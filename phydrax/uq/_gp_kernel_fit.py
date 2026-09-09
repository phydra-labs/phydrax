#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native exact-marginal-likelihood fitting for GP likelihood states."""

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..optim import (
    AbstractMinimizationMethod,
    MinimizationResult,
    minimize,
    OptimizationTermination,
)
from ._gp_backend import exact_gp_cholesky
from ._gp_likelihood import GaussianProcessLikelihoodState
from ._gp_multioutput import (
    MultiOutputGaussianProcessDiscrepancy,
    MultiOutputGaussianProcessLikelihoodState,
)
from ._posterior import ParameterSpace


class GaussianProcessKernelFitPolicy(StrictModule):
    """Exact GP state parameter space and explicit native optimizer policy."""

    parameter_space: ParameterSpace
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    minimum_data_count: int = eqx.field(static=True)
    refit_interval: int = eqx.field(static=True)

    def __init__(
        self,
        parameter_space: ParameterSpace,
        method: AbstractMinimizationMethod,
        /,
        *,
        termination: OptimizationTermination | None = None,
        minimum_data_count: int = 4,
        refit_interval: int = 1,
    ):
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        constrained = parameter_space.constrain(parameter_space.initial)
        if not isinstance(constrained, GaussianProcessLikelihoodState):
            raise TypeError(
                "parameter_space constrained values must be "
                "GaussianProcessLikelihoodState objects."
            )
        if not isinstance(method, AbstractMinimizationMethod):
            raise TypeError("method must implement AbstractMinimizationMethod.")
        resolved_termination = (
            OptimizationTermination() if termination is None else termination
        )
        if not isinstance(resolved_termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination or None.")
        self.parameter_space = parameter_space
        self.method = method
        self.termination = resolved_termination
        self.minimum_data_count = _positive_integer(
            minimum_data_count, name="minimum_data_count"
        )
        self.refit_interval = _positive_integer(refit_interval, name="refit_interval")


class GaussianProcessKernelFitResult(StrictModule):
    """Accepted state or explicitly retained previous epoch after failed fitting."""

    state: GaussianProcessLikelihoodState
    proposed_state: GaussianProcessLikelihoodState
    optimization: MinimizationResult
    accepted: Array
    previous_objective: Array
    proposed_objective: Array


class MultiOutputGaussianProcessKernelFitPolicy(StrictModule):
    """Multi-output GP parameter space and explicit native optimizer policy."""

    parameter_space: ParameterSpace
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    minimum_data_count: int = eqx.field(static=True)
    refit_interval: int = eqx.field(static=True)

    def __init__(
        self,
        parameter_space: ParameterSpace,
        method: AbstractMinimizationMethod,
        /,
        *,
        termination: OptimizationTermination | None = None,
        minimum_data_count: int = 4,
        refit_interval: int = 1,
    ):
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        constrained = parameter_space.constrain(parameter_space.initial)
        if not isinstance(constrained, MultiOutputGaussianProcessLikelihoodState):
            raise TypeError(
                "parameter_space constrained values must be "
                "MultiOutputGaussianProcessLikelihoodState objects."
            )
        if not isinstance(method, AbstractMinimizationMethod):
            raise TypeError("method must implement AbstractMinimizationMethod.")
        resolved_termination = (
            OptimizationTermination() if termination is None else termination
        )
        if not isinstance(resolved_termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination or None.")
        self.parameter_space = parameter_space
        self.method = method
        self.termination = resolved_termination
        self.minimum_data_count = _positive_integer(
            minimum_data_count, name="minimum_data_count"
        )
        self.refit_interval = _positive_integer(refit_interval, name="refit_interval")


class MultiOutputGaussianProcessKernelFitResult(StrictModule):
    """Accepted multi-output state or explicitly retained previous state."""

    state: MultiOutputGaussianProcessLikelihoodState
    proposed_state: MultiOutputGaussianProcessLikelihoodState
    optimization: MinimizationResult
    accepted: Array
    previous_objective: Array
    proposed_objective: Array


def fit_gaussian_process_kernel(
    points: ArrayLike,
    values: ArrayLike,
    policy: GaussianProcessKernelFitPolicy,
    /,
    *,
    previous_state: GaussianProcessLikelihoodState | None = None,
) -> GaussianProcessKernelFitResult:
    """Optimize exact negative log marginal likelihood plus prior/Jacobian."""
    if not isinstance(policy, GaussianProcessKernelFitPolicy):
        raise TypeError("policy must be a GaussianProcessKernelFitPolicy.")
    design = jnp.asarray(points)
    observations = jnp.asarray(values)
    if design.ndim < 2 or int(design.shape[0]) < policy.minimum_data_count:
        raise ValueError("Kernel fitting requires the declared minimum data count.")
    if observations.shape != (design.shape[0],):
        raise ValueError("values must align with points.")
    if not jnp.issubdtype(design.dtype, jnp.floating) or not jnp.issubdtype(
        observations.dtype, jnp.floating
    ):
        raise TypeError("Kernel fitting requires real floating-point data.")

    def objective(position, _):
        state = policy.parameter_space.constrain(position)
        likelihood = _negative_log_marginal_likelihood(design, observations, state)
        return likelihood - policy.parameter_space.unconstrained_log_prior(position)

    optimization = minimize(
        objective,
        policy.parameter_space.initial,
        method=policy.method,
        termination=policy.termination,
    )
    proposed = policy.parameter_space.constrain(optimization.parameters)
    previous = (
        policy.parameter_space.constrain(policy.parameter_space.initial)
        if previous_state is None
        else previous_state
    )
    previous_objective = _negative_log_marginal_likelihood(design, observations, previous)
    proposed_objective = _negative_log_marginal_likelihood(design, observations, proposed)
    finite = jnp.isfinite(proposed_objective)
    accepted = (
        optimization.successful & finite & (proposed_objective <= previous_objective)
    )
    state = _select_fit_state(proposed, previous, accepted)
    return GaussianProcessKernelFitResult(
        state=state,
        proposed_state=proposed,
        optimization=optimization,
        accepted=accepted,
        previous_objective=previous_objective,
        proposed_objective=proposed_objective,
    )


def fit_multioutput_gaussian_process_kernel(
    discrepancy: MultiOutputGaussianProcessDiscrepancy,
    policy: MultiOutputGaussianProcessKernelFitPolicy,
    /,
    *,
    physical_mean: ArrayLike | None = None,
    previous_state: MultiOutputGaussianProcessLikelihoodState | None = None,
) -> MultiOutputGaussianProcessKernelFitResult:
    """Fit a heterotopic multi-output GP by exact marginal likelihood."""
    if not isinstance(discrepancy, MultiOutputGaussianProcessDiscrepancy):
        raise TypeError("discrepancy must be a MultiOutputGaussianProcessDiscrepancy.")
    if not isinstance(policy, MultiOutputGaussianProcessKernelFitPolicy):
        raise TypeError("policy must be a MultiOutputGaussianProcessKernelFitPolicy.")
    if discrepancy.design.num_observations < policy.minimum_data_count:
        raise ValueError("Kernel fitting requires the declared minimum data count.")
    mean = (
        jnp.zeros_like(discrepancy.observations)
        if physical_mean is None
        else discrepancy.design.flatten(
            physical_mean,
            name="multi-output physical mean",
        )
    )

    def objective(position, _):
        state = policy.parameter_space.constrain(position)
        likelihood = -discrepancy.log_marginal_likelihood(mean, state=state)
        return likelihood - policy.parameter_space.unconstrained_log_prior(position)

    optimization = minimize(
        objective,
        policy.parameter_space.initial,
        method=policy.method,
        termination=policy.termination,
    )
    proposed = policy.parameter_space.constrain(optimization.parameters)
    previous = (
        policy.parameter_space.constrain(policy.parameter_space.initial)
        if previous_state is None
        else previous_state
    )
    if not isinstance(
        proposed, MultiOutputGaussianProcessLikelihoodState
    ) or not isinstance(previous, MultiOutputGaussianProcessLikelihoodState):
        raise TypeError(
            "Fitted states must be MultiOutputGaussianProcessLikelihoodState."
        )
    previous_objective = -discrepancy.log_marginal_likelihood(mean, state=previous)
    proposed_objective = -discrepancy.log_marginal_likelihood(mean, state=proposed)
    finite = jnp.isfinite(proposed_objective)
    accepted = (
        optimization.successful & finite & (proposed_objective <= previous_objective)
    )
    state = _select_fit_state(proposed, previous, accepted)
    return MultiOutputGaussianProcessKernelFitResult(
        state=state,
        proposed_state=proposed,
        optimization=optimization,
        accepted=accepted,
        previous_objective=previous_objective,
        proposed_objective=proposed_objective,
    )


def _select_fit_state(proposed, previous, accepted, /):
    return jax.tree_util.tree_map(
        lambda proposed_leaf, previous_leaf: jnp.where(
            accepted,
            proposed_leaf,
            previous_leaf,
        ),
        proposed,
        previous,
    )


def _negative_log_marginal_likelihood(
    points: Array,
    values: Array,
    state: GaussianProcessLikelihoodState,
    /,
) -> Array:
    if not isinstance(state, GaussianProcessLikelihoodState):
        raise TypeError("Fitted state must be a GaussianProcessLikelihoodState.")
    cholesky = exact_gp_cholesky(
        points,
        kernel=state.kernel,
        noise_scale=state.noise_scale,
        jitter=state.jitter,
    )
    whitened = jsp.linalg.solve_triangular(cholesky, values, lower=True)
    return (
        0.5 * jnp.vdot(whitened, whitened).real
        + jnp.sum(jnp.log(jnp.diag(cholesky)))
        + 0.5 * values.shape[0] * jnp.log(2.0 * jnp.pi)
    )


def _positive_integer(value: int, /, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = [
    "GaussianProcessKernelFitPolicy",
    "MultiOutputGaussianProcessKernelFitPolicy",
    "MultiOutputGaussianProcessKernelFitResult",
    "fit_multioutput_gaussian_process_kernel",
    "GaussianProcessKernelFitResult",
    "fit_gaussian_process_kernel",
]
