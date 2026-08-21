#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ..stochastic import StateSpaceProblem
from ._posterior import ParameterSpace
from ._state_space_path_density import state_space_path_log_density


class ParameterizedStateSpacePathLogDensity(StrictModule):
    """Global-parameter and latent-path normalized density decomposition."""

    parameter_prior: Array
    parameter_log_abs_det_jacobian: Array
    initial: Array
    transition: Array
    observation: Array
    case_log_density: Array
    log_density: Array
    valid: Array
    bound_problem: StateSpaceProblem
    approximation_id: str = eqx.field(static=True)


class ParameterizedStateSpaceProblem(StrictModule):
    """Bind unconstrained global parameters into an existing state-space schedule.

    Transition and observation implementations receive the output of ``bind_args``
    through ``StateSpaceStepContext.args``. The existing state-space model remains
    the sole simulation and likelihood hierarchy.
    """

    problem: StateSpaceProblem
    parameter_space: ParameterSpace
    bind_args_function: Callable[[PyTree[Any], Any], Any]
    initial_log_prob_function: Callable[[PyTree[Any], Array], Array] | None
    parameterization_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: StateSpaceProblem,
        parameter_space: ParameterSpace,
        bind_args: Callable[[PyTree[Any], Any], Any],
        /,
        *,
        initial_log_prob: Callable[[PyTree[Any], Array], Array] | None = None,
        parameterization_id: str = "parameterized-state-space",
    ):
        if not isinstance(problem, StateSpaceProblem):
            raise TypeError("problem must be a StateSpaceProblem.")
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        if not callable(bind_args):
            raise TypeError("bind_args must be callable.")
        if initial_log_prob is not None and not callable(initial_log_prob):
            raise TypeError("initial_log_prob must be callable or None.")
        identifier = str(parameterization_id)
        if not identifier:
            raise ValueError("parameterization_id must be non-empty.")
        self.problem = problem
        self.parameter_space = parameter_space
        self.bind_args_function = bind_args
        self.initial_log_prob_function = initial_log_prob
        self.parameterization_id = identifier

    @property
    def initial_position(self) -> PyTree[Any]:
        return self.parameter_space.initial

    def bind_physical(self, physical_parameters: PyTree[Any], /) -> StateSpaceProblem:
        bound_args = self.bind_args_function(physical_parameters, self.problem.args)
        return eqx.tree_at(lambda problem: problem.args, self.problem, bound_args)

    def bind(self, position: PyTree[Any], /) -> StateSpaceProblem:
        return self.bind_physical(self.parameter_space.constrain(position))

    def path_log_density(
        self,
        position: PyTree[Any],
        states: Array,
        /,
    ) -> ParameterizedStateSpacePathLogDensity:
        physical = self.parameter_space.constrain(position)
        bound_problem = self.bind_physical(physical)
        path = state_space_path_log_density(bound_problem, states)
        if self.initial_log_prob_function is None:
            initial = path.prior
        else:
            initial_states = jnp.take(
                jnp.asarray(states),
                0,
                axis=len(bound_problem.observations.case_shape),
            )
            initial = jnp.asarray(
                self.initial_log_prob_function(physical, initial_states)
            )
            if initial.shape != bound_problem.observations.case_shape:
                raise ValueError(
                    "initial_log_prob must return one scalar per physical case."
                )
        finite_initial = jnp.isfinite(initial)
        valid = path.valid & finite_initial
        case_log_density = jnp.where(
            valid,
            initial + jnp.sum(path.transition + path.observation, axis=-1),
            -jnp.inf,
        )
        parameter_prior = self.parameter_space.log_prior(physical)
        parameter_jacobian = self.parameter_space.log_abs_det_jacobian(position)
        log_density = parameter_prior + parameter_jacobian + jnp.sum(case_log_density)
        return ParameterizedStateSpacePathLogDensity(
            parameter_prior=parameter_prior,
            parameter_log_abs_det_jacobian=parameter_jacobian,
            initial=initial,
            transition=path.transition,
            observation=path.observation,
            case_log_density=case_log_density,
            log_density=log_density,
            valid=valid,
            bound_problem=bound_problem,
            approximation_id=self.parameterization_id,
        )


__all__ = [
    "ParameterizedStateSpacePathLogDensity",
    "ParameterizedStateSpaceProblem",
]
