#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ._belief_propagation import SumProductBeliefPropagationResult
from ._elimination import (
    plan_variable_elimination,
    variable_elimination,
    VariableEliminationMethod,
    VariableEliminationPlan,
    VariableEliminationResult,
)
from ._gibbs import (
    GibbsSchedule,
    GibbsState,
    PreparedChromaticGibbs,
    refresh_chromatic_gibbs,
    sample_gibbs,
)
from ._kernel import FactorGraphResourcePolicy
from ._model import DiscreteFactorGraph, factor_graph_log_score, pack_assignments
from ._training import contrastive_divergence_loss, FactorGraphTrainingDiagnostics


class PersistentFactorGraphTrainingState(StrictModule):
    """Optimizer and persistent negative-chain state for stochastic model learning."""

    graph: DiscreteFactorGraph
    optimizer_state: Any
    chains: GibbsState
    step_index: Array


class PersistentTrainingResult(StrictModule):
    state: PersistentFactorGraphTrainingState
    objective: Array
    diagnostics: FactorGraphTrainingDiagnostics
    sampler_valid: Array


class ExpectationMaximizationResult(StrictModule):
    graph: DiscreteFactorGraph
    posterior: VariableEliminationResult
    objective_before: Array
    objective_after: Array
    monotone: Array


def pseudolikelihood_loss(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    /,
) -> Array:
    """Return exact mean negative log scalar-conditional pseudolikelihood."""
    states = pack_assignments(graph, assignments)
    if states.ndim == 1:
        states = states[None, :]
    losses = []
    for variable, cardinality in enumerate(graph.cardinalities.tolist()):
        candidates = []
        for state in range(int(cardinality)):
            replaced = states.at[:, variable].set(state)
            candidates.append(factor_graph_log_score(graph, replaced))
        logits = jnp.stack(candidates, axis=-1)
        observed = jnp.take_along_axis(
            logits,
            states[:, variable, None],
            axis=-1,
        )[:, 0]
        losses.append(jax.nn.logsumexp(logits, axis=-1) - observed)
    return jnp.mean(jnp.stack(losses, axis=-1)) if losses else jnp.asarray(0.0)


def bethe_negative_log_likelihood(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    inference: SumProductBeliefPropagationResult,
    /,
) -> tuple[Array, FactorGraphTrainingDiagnostics]:
    """Return explicitly approximate Bethe negative log likelihood."""
    if not isinstance(inference, SumProductBeliefPropagationResult):
        raise TypeError("inference must be SumProductBeliefPropagationResult.")
    if inference.log_normalizer_kind != "bethe":
        raise ValueError("bethe_negative_log_likelihood requires a loopy Bethe result.")
    if inference.provenance.structure_id != graph.structure_id:
        raise ValueError("inference must describe the supplied graph structure.")
    states = pack_assignments(graph, assignments)
    if states.ndim == 1:
        states = states[None, :]
    scores = factor_graph_log_score(graph, states)
    objective = inference.log_normalizer - jnp.mean(scores)
    return objective, FactorGraphTrainingDiagnostics(
        objective=objective,
        positive_mean_log_score=jnp.mean(scores),
        negative_mean_log_score=jnp.asarray(jnp.nan, dtype=scores.dtype),
        positive_finite_fraction=jnp.mean(jnp.isfinite(scores)),
        negative_finite_fraction=jnp.asarray(1.0, dtype=scores.dtype),
        exact_normalizer=False,
    )


def initialize_persistent_training(
    graph: DiscreteFactorGraph,
    optimizer: optax.GradientTransformation,
    chains: GibbsState,
    /,
) -> PersistentFactorGraphTrainingState:
    """Initialize optimizer state over only trainable inexact graph leaves."""
    parameters = eqx.filter(graph, eqx.is_inexact_array)
    return PersistentFactorGraphTrainingState(
        graph=graph,
        optimizer_state=optimizer.init(parameters),
        chains=chains,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
    )


def persistent_contrastive_divergence_step(
    state: PersistentFactorGraphTrainingState,
    optimizer: optax.GradientTransformation,
    prepared: PreparedChromaticGibbs,
    positive_assignments: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    negative_sweeps: int = 1,
) -> PersistentTrainingResult:
    """Apply one persistent-CD/SML parameter update and advance negative chains."""
    if not isinstance(state, PersistentFactorGraphTrainingState):
        raise TypeError("state must be PersistentFactorGraphTrainingState.")
    if negative_sweeps < 1:
        raise ValueError("negative_sweeps must be positive.")
    if prepared.graph.structure_id != state.graph.structure_id:
        raise ValueError("prepared Gibbs plan must match the training graph.")
    if state.chains.positions.shape[1:] != (state.graph.num_variables,):
        raise ValueError("Persistent chains must match the training graph.")

    def objective(graph):
        return contrastive_divergence_loss(
            graph,
            positive_assignments,
            state.chains.positions,
        )

    (value, diagnostics), gradients = eqx.filter_value_and_grad(
        objective,
        has_aux=True,
    )(state.graph)
    parameters = eqx.filter(state.graph, eqx.is_inexact_array)
    updates, optimizer_state = optimizer.update(
        gradients,
        state.optimizer_state,
        parameters,
    )
    graph = eqx.apply_updates(state.graph, updates)
    refreshed = refresh_chromatic_gibbs(prepared, graph)
    sampled = sample_gibbs(
        refreshed,
        state.chains,
        key=key,
        schedule=GibbsSchedule(
            warmup_sweeps=0,
            num_draws=1,
            sweeps_per_draw=negative_sweeps,
        ),
    )
    next_state = PersistentFactorGraphTrainingState(
        graph=graph,
        optimizer_state=optimizer_state,
        chains=sampled.final_state,
        step_index=state.step_index + 1,
    )
    return PersistentTrainingResult(
        state=next_state,
        objective=value,
        diagnostics=diagnostics,
        sampler_valid=jnp.all(sampled.transition_valid),
    )


def stochastic_maximum_likelihood_step(
    state: PersistentFactorGraphTrainingState,
    optimizer: optax.GradientTransformation,
    prepared: PreparedChromaticGibbs,
    positive_assignments: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    negative_sweeps: int = 1,
) -> PersistentTrainingResult:
    """Alias the mathematically identical persistent-chain SML update contract."""
    return persistent_contrastive_divergence_step(
        state,
        optimizer,
        prepared,
        positive_assignments,
        key,
        negative_sweeps=negative_sweeps,
    )


def expectation_maximization_step(
    graph: DiscreteFactorGraph,
    plan: VariableEliminationPlan,
    m_step: Callable[
        [DiscreteFactorGraph, VariableEliminationResult], DiscreteFactorGraph
    ],
    /,
    *,
    evidence: ArrayLike,
) -> ExpectationMaximizationResult:
    """Run one exact E-step and caller-defined complete M-step with monotonicity evidence."""
    if not callable(m_step):
        raise TypeError("m_step must be callable.")
    if plan.graph.structure_id != graph.structure_id:
        raise ValueError("plan must describe the supplied graph structure.")
    posterior = variable_elimination(plan, evidence=evidence)
    objective_before = posterior.log_normalizer
    updated = m_step(graph, posterior)
    if not isinstance(updated, DiscreteFactorGraph):
        raise TypeError("m_step must return DiscreteFactorGraph.")
    if updated.structure_id != graph.structure_id:
        raise ValueError("m_step must preserve the factor-graph structure.")
    updated_plan = plan_variable_elimination(
        updated,
        VariableEliminationMethod(ordering="given", order=plan.order),
        resources=FactorGraphResourcePolicy(
            maximum_elimination_elements=max(
                plan.maximum_workspace_elements,
                1,
            ),
            maximum_treewidth=max(plan.treewidth, 1),
        ),
    )
    after = variable_elimination(updated_plan, evidence=evidence)
    return ExpectationMaximizationResult(
        graph=updated,
        posterior=posterior,
        objective_before=objective_before,
        objective_after=after.log_normalizer,
        monotone=after.log_normalizer >= objective_before,
    )


__all__ = [
    "ExpectationMaximizationResult",
    "PersistentFactorGraphTrainingState",
    "PersistentTrainingResult",
    "bethe_negative_log_likelihood",
    "expectation_maximization_step",
    "initialize_persistent_training",
    "persistent_contrastive_divergence_step",
    "pseudolikelihood_loss",
    "stochastic_maximum_likelihood_step",
]
