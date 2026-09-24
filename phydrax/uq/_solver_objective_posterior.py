#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Posterior problems whose likelihood is a residual-valued solver objective."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._admissibility import guard_derivative_validity
from .._trainable import combine_parameters, partition_parameters
from ..solver._solver_objective import RolloutObjective, SolverObjective
from ._posterior import ParameterSpace, PosteriorProblem


def posterior_problem_from_solver_objective(
    objective: SolverObjective | RolloutObjective,
    tree: PyTree[Any],
    parameter_space: ParameterSpace,
    /,
) -> PosteriorProblem:
    """Posterior over a component's parameters with a solver-objective likelihood.

    The position is the flat PARAMETER lane of the objective's selected
    component (`jax.flatten_util.ravel_pytree` order), so `parameter_space` must
    describe one real vector of that size; its priors and bijectors stay the
    caller's explicit choice. Every case must be residual-valued, with the
    residual whitened by its observation noise: the log likelihood is
    `-0.5 * ||r||^2` over the concatenated case residuals, and
    `gauss_newton_residual` returns `r`.

    Failure reduction is fail-closed: a failed or nonfinite case makes the whole
    residual not a number and the log likelihood `-inf`, so EKI raises and
    samplers reject the position; a likelihood never drops data. The objective
    must therefore use `accepted_results="reject-attempt"`, and every PARAMETER
    leaf of the component must be admitted by its `(route, kind)`.

    EKI (`fit_eki`) and distribution-evolution training are the derivative-free
    consumers and are always admitted. When a trained model's derivative
    contract does not admit the objective's route, any differentiation of the
    likelihood or residual raises instead of returning a derivative; nothing
    switches between gradient-based and derivative-free consumers silently.
    """
    if not isinstance(objective, (SolverObjective, RolloutObjective)):
        raise TypeError(
            "objective must be a SolverObjective or RolloutObjective; algorithmic "
            "work is not a likelihood."
        )
    if objective.accepted_results != "reject-attempt":
        raise ValueError(
            f"{objective.context}: a likelihood cannot drop failed cases; use "
            "accepted_results='reject-attempt'."
        )
    if not isinstance(parameter_space, ParameterSpace):
        raise TypeError("parameter_space must be a ParameterSpace.")
    admission = objective.admit(tree, differentiable=False)
    if admission.stopped:
        raise ValueError(
            f"{objective.context}: the posterior position would include parameters "
            f"{tuple(path for path, _ in admission.stopped)!r} whose authority does "
            "not admit this objective; select only the admitted component."
        )
    parameters, model_state, fixed = partition_parameters(objective.select(tree))
    flat, unravel = ravel_pytree(parameters)
    if parameter_space.physical_shapes != ((flat.size,),):
        raise ValueError(
            f"{objective.context}: parameter_space must describe one vector of the "
            f"component's {flat.size} parameters."
        )
    failures = admission.derivative_failures
    message = (
        f"{objective.context}: the likelihood is not differentiable "
        f"({'; '.join(failures)}); use a derivative-free consumer such as fit_eki."
    )

    def guarded(value: Any) -> Any:
        if not failures:
            return value
        return guard_derivative_validity(value, False, failure="error", message=message)

    def residual(position: Array) -> Array:
        component = combine_parameters(
            unravel(jnp.asarray(position, flat.dtype)), model_state, fixed
        )
        vectors, accepted = objective._residual_cases(component)
        return guarded(
            jnp.where(jnp.all(accepted), jnp.ravel(vectors), jnp.asarray(jnp.nan))
        )

    def log_likelihood(position: Array) -> Array:
        values = residual(position)
        finite = jnp.all(jnp.isfinite(values))
        safe = jnp.where(finite, values, jnp.zeros_like(values))
        return jnp.where(finite, -0.5 * jnp.sum(safe * safe), -jnp.inf)

    return PosteriorProblem(
        parameter_space, log_likelihood, gauss_newton_residual=residual
    )


__all__ = ["posterior_problem_from_solver_objective"]
