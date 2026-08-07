#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._posterior import PosteriorProblem


class MAPResult(StrictModule):
    """Result and convergence evidence for posterior mode optimization."""

    problem: PosteriorProblem
    position: PyTree[Array]
    parameters: PyTree[Array]
    gradient: PyTree[Array]
    objective: Array
    log_density: Array
    gradient_norm: Array
    objective_history: Array
    num_steps: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    initial_compilation_seconds: float = eqx.field(static=True)
    initial_evaluation_seconds: float = eqx.field(static=True)
    step_compilation_seconds: float = eqx.field(static=True)
    optimization_seconds: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        position: PyTree[Array],
        gradient: PyTree[Array],
        objective: Array,
        gradient_norm: Array,
        objective_history: Array,
        num_steps: int,
        objective_evaluations: int,
        converged: bool,
        termination_reason: str,
        duration_seconds: float,
        initial_compilation_seconds: float,
        initial_evaluation_seconds: float,
        step_compilation_seconds: float,
        optimization_seconds: float,
    ):
        self.problem = problem
        self.position = position
        self.parameters = problem.parameter_space.constrain(position)
        self.gradient = gradient
        self.objective = jnp.asarray(objective)
        self.log_density = -jnp.asarray(objective)
        self.gradient_norm = jnp.asarray(gradient_norm)
        self.objective_history = jnp.asarray(objective_history)
        self.num_steps = int(num_steps)
        self.objective_evaluations = int(objective_evaluations)
        self.converged = bool(converged)
        self.termination_reason = str(termination_reason)
        self.duration_seconds = float(duration_seconds)
        self.initial_compilation_seconds = float(initial_compilation_seconds)
        self.initial_evaluation_seconds = float(initial_evaluation_seconds)
        self.step_compilation_seconds = float(step_compilation_seconds)
        self.optimization_seconds = float(optimization_seconds)

    @property
    def compilation_seconds(self) -> float:
        """Total initial-evaluation and optimizer-step compilation time."""
        return self.initial_compilation_seconds + self.step_compilation_seconds

    @property
    def execution_seconds(self) -> float:
        """Initial objective evaluation plus compiled optimizer-step execution."""
        return self.initial_evaluation_seconds + self.optimization_seconds

    @property
    def mean_step_seconds(self) -> float:
        """Mean compiled optimizer transition time, or zero when already converged."""
        if self.num_steps == 0:
            return 0.0
        return self.optimization_seconds / self.num_steps


class MAPConvergenceError(RuntimeError):
    """Raised when MAP optimization terminates without satisfying its gradient gate."""

    result: MAPResult

    def __init__(self, result: MAPResult):
        self.result = result
        super().__init__(
            "MAP optimization did not converge: "
            f"reason={result.termination_reason!r}, "
            f"steps={result.num_steps}, "
            f"gradient_norm={float(result.gradient_norm):.6g}."
        )


@eqx.filter_jit
def _compiled_initial_evaluation(
    position: PyTree[Array],
    problem: PosteriorProblem,
) -> tuple[Array, PyTree[Array], Array]:
    objective_fn = lambda value: problem.negative_log_density(value)
    objective, gradient = jax.value_and_grad(objective_fn)(position)
    return objective, gradient, _tree_l2_norm(gradient)


@eqx.filter_jit
def _compiled_lbfgs_step(
    position: PyTree[Array],
    optimizer_state: Any,
    objective: Array,
    gradient: PyTree[Array],
    problem: PosteriorProblem,
    *,
    learning_rate: float,
    memory: int,
) -> tuple[PyTree[Array], Any, Array, PyTree[Array], Array]:
    objective_fn = lambda value: problem.negative_log_density(value)
    optimizer = optax.lbfgs(
        learning_rate=learning_rate,
        memory_size=memory,
    )
    value_and_grad_from_state = optax.value_and_grad_from_state(objective_fn)
    updates, next_optimizer_state = optimizer.update(
        gradient,
        optimizer_state,
        position,
        value=objective,
        grad=gradient,
        value_fn=objective_fn,
    )
    next_position = eqx.apply_updates(position, updates)
    next_objective, next_gradient = value_and_grad_from_state(
        next_position,
        state=next_optimizer_state,
    )
    next_objective = jnp.asarray(next_objective, dtype=float).reshape(())
    return (
        next_position,
        next_optimizer_state,
        next_objective,
        next_gradient,
        _tree_l2_norm(next_gradient),
    )


def find_map(
    problem: PosteriorProblem,
    initial_position: PyTree[Array] | None = None,
    /,
    *,
    max_steps: int = 500,
    gradient_tolerance: float = 1e-6,
    objective_tolerance: float | None = None,
    learning_rate: float = 1.0,
    memory: int = 10,
    raise_on_failure: bool = True,
) -> MAPResult:
    """Minimize a deterministic posterior's negative log density with L-BFGS."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    steps_limit = int(max_steps)
    if steps_limit <= 0:
        raise ValueError("max_steps must be positive.")
    gradient_gate = float(gradient_tolerance)
    if not jnp.isfinite(gradient_gate) or gradient_gate < 0.0:
        raise ValueError("gradient_tolerance must be finite and non-negative.")
    objective_gate = None if objective_tolerance is None else float(objective_tolerance)
    if objective_gate is not None and (
        not jnp.isfinite(objective_gate) or objective_gate < 0.0
    ):
        raise ValueError("objective_tolerance must be finite and non-negative or None.")
    step_size = float(learning_rate)
    if not jnp.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    history_size = int(memory)
    if history_size <= 0:
        raise ValueError("memory must be positive.")

    position = problem.initial_position if initial_position is None else initial_position
    problem.parameter_space.constrain(position)
    optimizer = optax.lbfgs(
        learning_rate=step_size,
        memory_size=history_size,
    )
    optimizer_state = optimizer.init(position)
    objective_evaluations = 1

    started = time.perf_counter()
    compilation_started = time.perf_counter()
    compiled_initial_evaluation = cast(Any, _compiled_initial_evaluation).lower(
        position,
        problem,
    ).compile()
    initial_compilation_seconds = time.perf_counter() - compilation_started
    evaluation_started = time.perf_counter()
    objective, gradient, gradient_norm = compiled_initial_evaluation(position, problem)
    jax.block_until_ready(gradient_norm)
    initial_evaluation_seconds = time.perf_counter() - evaluation_started
    objective = jnp.asarray(objective, dtype=float).reshape(())
    _validate_evaluation(objective, gradient)
    objective_history = [objective]
    converged = float(gradient_norm) <= gradient_gate
    termination_reason = "gradient_tolerance" if converged else "max_steps"
    num_steps = 0

    step_compilation_seconds = 0.0
    optimization_seconds = 0.0
    compiled_step = None
    if not converged:
        compilation_started = time.perf_counter()
        compiled_step = cast(Any, _compiled_lbfgs_step).lower(
            position,
            optimizer_state,
            objective,
            gradient,
            problem,
            learning_rate=step_size,
            memory=history_size,
        ).compile()
        step_compilation_seconds = time.perf_counter() - compilation_started

    for step in range(1, steps_limit + 1):
        if converged:
            break
        if compiled_step is None:
            raise RuntimeError("Compiled L-BFGS step is unavailable.")

        step_started = time.perf_counter()
        (
            next_position,
            optimizer_state,
            next_objective,
            next_gradient,
            next_gradient_norm,
        ) = compiled_step(
            position,
            optimizer_state,
            objective,
            gradient,
            problem,
            learning_rate=step_size,
            memory=history_size,
        )
        jax.block_until_ready(next_gradient_norm)
        optimization_seconds += time.perf_counter() - step_started
        objective_evaluations += _line_search_evaluations(optimizer_state)
        next_objective = jnp.asarray(next_objective, dtype=float).reshape(())
        num_steps = step

        try:
            _validate_evaluation(next_objective, next_gradient)
        except FloatingPointError:
            objective = next_objective
            gradient = next_gradient
            gradient_norm = next_gradient_norm
            position = next_position
            objective_history.append(next_objective)
            termination_reason = "non_finite_evaluation"
            break

        objective_history.append(next_objective)
        previous_value = float(jnp.asarray(objective, dtype=float).reshape(()))
        next_value = float(jnp.asarray(next_objective, dtype=float).reshape(()))
        position = next_position
        objective = next_objective
        gradient = next_gradient
        gradient_norm = next_gradient_norm

        if float(gradient_norm) <= gradient_gate:
            converged = True
            termination_reason = "gradient_tolerance"
            break
        if objective_gate is not None:
            scale = max(1.0, abs(previous_value), abs(next_value))
            if abs(previous_value - next_value) <= objective_gate * scale:
                termination_reason = "objective_stagnation"
                break

    jax.block_until_ready(objective)
    duration = time.perf_counter() - started
    result = MAPResult(
        problem=problem,
        position=position,
        gradient=gradient,
        objective=objective,
        gradient_norm=gradient_norm,
        objective_history=jnp.stack(tuple(objective_history)),
        num_steps=num_steps,
        objective_evaluations=objective_evaluations,
        converged=converged,
        termination_reason=termination_reason,
        duration_seconds=duration,
        initial_compilation_seconds=initial_compilation_seconds,
        initial_evaluation_seconds=initial_evaluation_seconds,
        step_compilation_seconds=step_compilation_seconds,
        optimization_seconds=optimization_seconds,
    )
    if not result.converged and raise_on_failure:
        raise MAPConvergenceError(result)
    return result


def _tree_l2_norm(tree: PyTree[Any]) -> Array:
    return jnp.sqrt(
        sum(
            (
                jnp.sum(jnp.abs(jnp.asarray(leaf)) ** 2)
                for leaf in jax.tree_util.tree_leaves(tree)
            ),
            jnp.zeros(()),
        )
    )


def _line_search_evaluations(optimizer_state: Any) -> int:
    line_search_state = optimizer_state[-1]
    info = getattr(line_search_state, "info", None)
    count = getattr(info, "num_linesearch_steps", 1)
    return max(1, int(count))


def _validate_evaluation(objective: Array, gradient: PyTree[Any]) -> None:
    if jnp.asarray(objective).ndim != 0 or not bool(jnp.isfinite(objective)):
        raise FloatingPointError("MAP objective must be a finite scalar.")
    if any(
        bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(gradient)
    ):
        raise FloatingPointError("MAP gradient must be finite.")


__all__ = ["MAPConvergenceError", "MAPResult", "find_map"]
