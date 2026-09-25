#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import combine_parameters
from .._training_kernel import (
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    run_training_attempt,
    SubspaceTrainingTree,
    TrainingKernelSpec,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
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
            (
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


_FACTOR_BATCH_ADDRESS = SampleAddress("uq.sing", "factor-batch", role="outer-iteration")
_FINAL_SMOOTHER_ADDRESS = SampleAddress(
    "uq.sing", "final-smoother", role="initialization"
)


def _negative_sing_objective(parameters, model_state, fixed, payload, keys):
    """Held-posterior negative SING objective of the subspace parameters.

    The payload is `(posterior_state, batch, transition_plan, observation_factor)`.
    """
    del keys
    posterior_state, batch, transition_plan, observation_factor = payload
    tree = combine_parameters(parameters, model_state, fixed)
    result = sing_objective(
        tree.model(),
        posterior_state,
        transition_plan=transition_plan,
        observation_factor=observation_factor,
        batch=batch,
    )
    return _ObjectiveContribution(-result.objective, 1.0), model_state, ()


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
    Every parameter step is one attempt of the shared training kernel (`MODEL`
    root authority, one data-fit objective) over the subspace selection. A
    nonfinite parameter step rolls back and ends learning with `valid=False`
    (`status=1`); the result then holds the last accepted parameters.
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
        kernel = None
        training = None
    else:
        if not isinstance(parameter_subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be ParameterSubspace or None.")
        parameter_subspace.validate_root(problem)
        position = parameter_subspace.pack()
        tree = SubspaceTrainingTree.from_subspace(parameter_subspace)
        kernel = prepare_training_kernel(
            tree,
            (
                KernelObjective(
                    objective_id="sing-held-posterior-objective",
                    kind=ObjectiveKind.DATA_FIT,
                    route=DerivativeRoute.DIRECT,
                    fn=_negative_sing_objective,
                ),
            ),
            TrainingKernelSpec(
                OptaxUpdateRule(optimizer, rule_id="sing-caller-optimizer"),
                context="fit_sing",
                rejection_budget=0,
            ),
            root_authority=ComponentAuthority.MODEL,
        )
        training = kernel.init(tree, key)
    objective_values = []
    audits = []
    factor_sampling_state = None
    objective_kind = "elbo"
    training_failed = False
    for outer in range(policy.max_outer_iterations):
        posterior = sing_smoother(
            current_problem,
            state=posterior.state,
            max_iterations=policy.posterior_steps,
        )
        if policy.factor_source is None:
            batch = None
        else:
            batch, factor_sampling_state = policy.factor_source(
                derive_key(key, _FACTOR_BATCH_ADDRESS, outer),
                outer,
                factor_sampling_state,
            )
        if kernel is not None:
            payload = (
                posterior.state,
                batch,
                policy.transition_plan,
                observation_factor,
            )
            for _ in range(policy.parameter_steps):
                try:
                    training, _ = run_training_attempt(kernel, training, payload)
                except TrainingRejectionBudgetError:
                    training_failed = True
                    break
            learned = kernel.tree(training)
            position = parameter_subspace.pack(learned.selected)
            current_problem = learned.model()
            if training_failed:
                break
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
    posterior = sing_smoother(
        current_problem,
        key=derive_key(key, _FINAL_SMOOTHER_ADDRESS),
        max_iterations=policy.posterior_steps,
    )
    final_audit = sing_objective(
        current_problem,
        posterior.state,
        transition_plan=policy.transition_plan,
        observation_factor=observation_factor,
        batch=None,
    )
    if training_failed:
        # Learning stopped early: the final audit is a new record of the last
        # accepted parameters, not a replacement of a completed iteration.
        audits.append(final_audit.objective)
        objective_values.append(final_audit.objective)
    else:
        audits[-1] = final_audit.objective
        objective_values[-1] = final_audit.objective
    objective_kind = final_audit.objective_kind
    history = jnp.stack(objective_values)
    full_history = jnp.stack(audits)
    valid = (
        posterior.valid
        & jnp.all(jnp.isfinite(history))
        & jnp.all(jnp.isfinite(full_history))
        & (full_history.size > 0)
        & (not training_failed)
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
