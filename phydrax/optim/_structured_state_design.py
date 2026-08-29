#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._iterative import (
    Bounds,
    MinimizationProblem,
    NonlinearConstraint,
    OptimizationTermination,
)
from ._pde_constrained import StateDesignProblem
from ._structured_compile import (
    compile_structured_minimization,
    solve_structured_minimization,
    StructuredMinimizationCompilation,
    StructuredMinimizationResult,
)
from ._structured_method import AbstractStructuredNonlinearMethod


class StructuredStateDesignCompilation(StrictModule):
    """All-at-once state/design problem lowered to one structured NLP."""

    problem: StateDesignProblem
    optimization: StructuredMinimizationCompilation


class StructuredStateDesignResult(StrictModule):
    """Decoded state/design pair and underlying structured optimization evidence."""

    state: PyTree[Array]
    design: PyTree[Array]
    objective: Array
    optimization: StructuredMinimizationResult

    @property
    def successful(self) -> Array:
        return self.optimization.successful


def compile_structured_state_design(
    problem: StateDesignProblem,
    initial_state: PyTree[Any],
    initial_design: PyTree[Any],
    /,
    *,
    sample_args: Any = None,
    exact_hessian: bool = True,
    compiler: Any = "auto",
    chunk_size: int | None = None,
) -> StructuredStateDesignCompilation:
    """Compile a fixed-topology all-at-once state/design constrained problem."""
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    state_lower = jax.tree.map(
        lambda value: jnp.full_like(value, -jnp.inf), initial_state
    )
    state_upper = jax.tree.map(lambda value: jnp.full_like(value, jnp.inf), initial_state)
    if problem.design_bounds is None:
        design_lower = jax.tree.map(
            lambda value: jnp.full_like(value, -jnp.inf),
            initial_design,
        )
        design_upper = jax.tree.map(
            lambda value: jnp.full_like(value, jnp.inf),
            initial_design,
        )
    else:
        design_lower, design_upper = problem.design_bounds.materialize(initial_design)
    sample_residual = problem.residual(initial_state, initial_design, sample_args)
    zeros = jax.tree.map(jnp.zeros_like, sample_residual)

    def objective(values, args):
        value, auxiliary = problem.value(values[0], values[1], args)
        return (value, auxiliary) if problem.has_aux else value

    minimization = MinimizationProblem(
        objective,
        has_aux=problem.has_aux,
        bounds=Bounds(
            (state_lower, design_lower),
            (state_upper, design_upper),
        ),
        constraints=(
            NonlinearConstraint(
                lambda values, args: problem.residual(values[0], values[1], args),
                lower=zeros,
                upper=zeros,
                constraint_id=f"{problem.problem_id}:state-equation",
            ),
        ),
        problem_id=f"{problem.problem_id}:structured-all-at-once",
    )
    compiled = compile_structured_minimization(
        minimization,
        (initial_state, initial_design),
        sample_args=sample_args,
        exact_hessian=exact_hessian,
        compiler=compiler,
        chunk_size=chunk_size,
    )
    return StructuredStateDesignCompilation(problem, compiled)


def solve_structured_state_design(
    compilation: StructuredStateDesignCompilation,
    /,
    *,
    method: AbstractStructuredNonlinearMethod,
    termination: OptimizationTermination | None = None,
    initial_state: PyTree[Any] | None = None,
    initial_design: PyTree[Any] | None = None,
    warm_start: Any = None,
) -> StructuredStateDesignResult:
    """Solve and decode one all-at-once structured state/design problem."""
    if not isinstance(compilation, StructuredStateDesignCompilation):
        raise TypeError("compilation must be a StructuredStateDesignCompilation.")
    initial = None
    if (initial_state is None) != (initial_design is None):
        raise ValueError("initial_state and initial_design must be supplied together.")
    if initial_state is not None and initial_design is not None:
        initial = (initial_state, initial_design)
    solved = solve_structured_minimization(
        compilation.optimization,
        method=method,
        termination=termination,
        initial_parameters=initial,
        warm_start=warm_start,
    )
    state, design = solved.optimization.parameters
    objective, _ = compilation.problem.value(
        state,
        design,
        compilation.optimization.prepared.args,
    )
    return StructuredStateDesignResult(state, design, objective, solved)


__all__ = [
    "StructuredStateDesignCompilation",
    "StructuredStateDesignResult",
    "compile_structured_state_design",
    "solve_structured_state_design",
]
