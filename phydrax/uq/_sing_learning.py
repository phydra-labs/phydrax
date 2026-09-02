#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..nn.parameters import ParameterSubspace
from ._sing import sing_smoother, SINGResult, SINGState
from ._sing_transition import sing_objective, SINGTransitionPlan


class SINGLearningPolicy(StrictModule):
    """Alternating fixed-posterior/fixed-model finite learning epochs."""

    transition_plan: SINGTransitionPlan
    factor_source: Any
    posterior_steps: int = eqx.field(static=True)
    parameter_steps: int = eqx.field(static=True)
    max_outer_iterations: int = eqx.field(static=True)
    full_audit_every: int = eqx.field(static=True)

    def __init__(
        self,
        posterior_steps: int,
        parameter_steps: int,
        max_outer_iterations: int,
        full_audit_every: int,
        transition_plan: SINGTransitionPlan,
        /,
        *,
        factor_source: Any = None,
    ):
        counts = tuple(
            int(value)
            for value in (
                posterior_steps,
                parameter_steps,
                max_outer_iterations,
                full_audit_every,
            )
        )
        if any(value <= 0 for value in counts):
            raise ValueError("all SING learning iteration counts must be positive.")
        if not isinstance(transition_plan, SINGTransitionPlan):
            raise TypeError("transition_plan must be a SINGTransitionPlan.")
        if factor_source is not None and not callable(factor_source):
            raise TypeError("factor_source must be callable or None.")
        (
            self.posterior_steps,
            self.parameter_steps,
            self.max_outer_iterations,
            self.full_audit_every,
        ) = counts
        self.transition_plan = transition_plan
        self.factor_source = factor_source


class SINGLearningResult(StrictModule):
    """Alternating-learning output with mandatory full-audit history."""

    problem: Any
    posterior: SINGResult
    learned_parameters: Array
    objective_history: Array
    full_audit_history: Array
    transition_evidence: tuple[str, ...] = eqx.field(static=True)
    factor_sampling_state: Any
    valid: Array
    status: Array
    objective_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def fit_sing(
    problem: Any,
    /,
    *,
    policy: SINGLearningPolicy,
    parameter_subspace: ParameterSubspace | None = None,
    optimizer: optax.GradientTransformation,
    state: SINGState | None = None,
    observation_factor: Any = None,
    key: Key[Array, ""],
) -> SINGLearningResult:
    """Alternate posterior natural steps and held-posterior parameter steps.

    Gradients are taken only through the fixed transition/support/factor route.
    Selection, support rank, and inducing topology changes require a new call.
    """
    if not isinstance(policy, SINGLearningPolicy):
        raise TypeError("policy must be a SINGLearningPolicy.")
    current_problem = problem
    posterior = sing_smoother(
        current_problem,
        state=state,
        key=key if state is None else None,
        max_iterations=policy.posterior_steps,
    )
    if parameter_subspace is None:
        if policy.parameter_steps != 1:
            raise ValueError(
                "parameter_subspace is required when parameter_steps requests learning."
            )
        position = jnp.zeros((0,), dtype=posterior.elbo.total_elbo.dtype)
        optimizer_state = None
    else:
        if not isinstance(parameter_subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be ParameterSubspace or None.")
        parameter_subspace.validate_root(problem)
        position = parameter_subspace.pack()
        optimizer_state = optimizer.init(position)
    objective_values = []
    audits = []
    factor_sampling_state = None
    objective_kind = "elbo"
    for outer in range(policy.max_outer_iterations):
        outer_key = jr.fold_in(key, outer)
        posterior = sing_smoother(
            current_problem,
            state=posterior.state,
            max_iterations=policy.posterior_steps,
        )
        if policy.factor_source is None:
            batch = None
        else:
            batch, factor_sampling_state = policy.factor_source(
                outer_key,
                outer,
                factor_sampling_state,
            )
        if parameter_subspace is not None:
            frozen_posterior = posterior.state

            def loss(vector):
                candidate = parameter_subspace.reconstruct_vector(vector)
                result = sing_objective(
                    candidate,
                    frozen_posterior,
                    transition_plan=policy.transition_plan,
                    observation_factor=observation_factor,
                    batch=batch,
                )
                return -result.objective

            for _ in range(policy.parameter_steps):
                _, gradient = jax.value_and_grad(loss)(position)
                updates, optimizer_state = optimizer.update(
                    gradient,
                    optimizer_state,
                    position,
                )
                position = optax.apply_updates(position, updates)
            current_problem = parameter_subspace.reconstruct_vector(position)
        represented = sing_objective(
            current_problem,
            posterior.state,
            transition_plan=policy.transition_plan,
            observation_factor=observation_factor,
            batch=batch,
        )
        objective_values.append(represented.objective)
        objective_kind = represented.objective_kind
        if (
            outer + 1
        ) % policy.full_audit_every == 0 or outer + 1 == policy.max_outer_iterations:
            audit = sing_objective(
                current_problem,
                posterior.state,
                transition_plan=policy.transition_plan,
                observation_factor=observation_factor,
                batch=None,
            )
            audits.append(audit.objective)
    history = jnp.stack(objective_values)
    full_history = jnp.stack(audits)
    valid = (
        posterior.valid
        & jnp.all(jnp.isfinite(history))
        & jnp.all(jnp.isfinite(full_history))
        & (full_history.size > 0)
    )
    status = jnp.where(valid, 0, 1).astype(jnp.int32)
    transition_evidence = (
        f"transition-plan:{policy.transition_plan.plan_id}",
        f"transition-method:{policy.transition_plan.method}",
        "parameter-gradient:fixed-posterior",
        "selection-gradient:none",
        "final-objective:full-audit",
    )
    return SINGLearningResult(
        problem=current_problem,
        posterior=posterior,
        learned_parameters=position,
        objective_history=history,
        full_audit_history=full_history,
        transition_evidence=transition_evidence,
        factor_sampling_state=factor_sampling_state,
        valid=valid,
        status=status,
        objective_kind=objective_kind,
        bounded_non_claim=(
            "Alternating optimization is local. Minibatch values are optimization "
            "estimators only; reported acceptance requires the retained full audits."
        ),
    )


__all__ = ["SINGLearningPolicy", "SINGLearningResult", "fit_sing"]
