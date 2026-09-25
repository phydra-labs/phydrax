#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Key

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._strict import StrictModule
from .._trainable import combine_parameters
from .._training_kernel import (
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    PreparedTrainingKernel,
    run_training_attempt,
    TrainingKernelSpec,
    TrainingKernelState,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
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
from ._training import (
    _packed_contrastive_divergence_loss,
    FactorGraphTrainingDiagnostics,
)


class PersistentFactorGraphTrainingState(StrictModule):
    """Committed model, training-kernel state, and persistent negative chains.

    `training` is the shared training-kernel state (parameters, optimizer state,
    attempt/accepted cursors); `graph` is its committed factor graph.
    """

    graph: DiscreteFactorGraph
    training: TrainingKernelState
    chains: GibbsState

    @property
    def step_index(self) -> Array:
        """Number of attempted persistent-CD updates."""
        return self.training.attempt_cursor


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
    log_normalizer = eqx.error_if(
        inference.log_normalizer,
        ~inference.successful | ~inference.converged,
        "Bethe likelihood requires converged successful inference.",
    )
    objective = log_normalizer - jnp.mean(scores)
    return objective, FactorGraphTrainingDiagnostics(
        objective=objective,
        positive_mean_log_score=jnp.mean(scores),
        negative_mean_log_score=jnp.asarray(jnp.nan, dtype=scores.dtype),
        positive_finite_fraction=jnp.mean(jnp.isfinite(scores)),
        negative_finite_fraction=jnp.asarray(1.0, dtype=scores.dtype),
        exact_normalizer=False,
    )


def _contrastive_divergence_objective(parameters, model_state, fixed, payload, keys):
    """Persistent-CD surrogate; the payload is packed `(positives, negatives)`."""
    del keys
    positive, negative = payload
    value, diagnostics = _packed_contrastive_divergence_loss(
        combine_parameters(parameters, model_state, fixed), positive, negative
    )
    return _ObjectiveContribution(value, 1.0), model_state, diagnostics


def _persistent_kernel(
    graph: DiscreteFactorGraph,
    optimizer: optax.GradientTransformation,
    /,
    *,
    context: str,
) -> PreparedTrainingKernel:
    return prepare_training_kernel(
        graph,
        (
            KernelObjective(
                objective_id="persistent-contrastive-divergence",
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_contrastive_divergence_objective,
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(optimizer, rule_id="persistent-cd-caller-optimizer"),
            context=context,
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )


# The contrastive-divergence objective draws no training randomness (negative
# chains advance with the caller's per-step key), so the kernel root key only
# completes the checkpointable state.
_INERT_ROOT_KEY_SEED = 0


def initialize_persistent_training(
    graph: DiscreteFactorGraph,
    optimizer: optax.GradientTransformation,
    chains: GibbsState,
    /,
) -> PersistentFactorGraphTrainingState:
    """Initialize the training-kernel state over the graph's PARAMETER leaves."""
    kernel = _persistent_kernel(
        graph, optimizer, context="initialize_persistent_training"
    )
    return PersistentFactorGraphTrainingState(
        graph=graph,
        training=kernel.init(graph, jax.random.key(_INERT_ROOT_KEY_SEED)),
        chains=chains,
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
    """Apply one persistent-CD/SML parameter update and advance negative chains.

    The update is one attempt of the shared training kernel (`MODEL` root
    authority, one data-fit objective). A nonfinite update rolls back and raises
    `FloatingPointError`; the chains then do not advance.
    """
    if not isinstance(state, PersistentFactorGraphTrainingState):
        raise TypeError("state must be PersistentFactorGraphTrainingState.")
    if negative_sweeps < 1:
        raise ValueError("negative_sweeps must be positive.")
    if prepared.graph.structure_id != state.graph.structure_id:
        raise ValueError("prepared Gibbs plan must match the training graph.")
    if state.chains.positions.shape[1:] != (state.graph.num_variables,):
        raise ValueError("Persistent chains must match the training graph.")
    kernel = _persistent_kernel(
        state.graph, optimizer, context="persistent_contrastive_divergence_step"
    )
    try:
        training, evidence = run_training_attempt(
            kernel,
            state.training,
            (
                pack_assignments(state.graph, positive_assignments),
                pack_assignments(state.graph, state.chains.positions),
            ),
        )
    except TrainingRejectionBudgetError as error:
        raise FloatingPointError(
            "Persistent contrastive-divergence update is nonfinite; the state was "
            "rolled back and the chains did not advance."
        ) from error
    graph = kernel.tree(training)
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
        training=training,
        chains=sampled.final_state,
    )
    return PersistentTrainingResult(
        state=next_state,
        objective=evidence.diagnostics[0].objective,
        diagnostics=evidence.diagnostics[0],
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
