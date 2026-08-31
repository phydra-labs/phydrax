#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..graph import segment_sum
from ._belief_propagation import _broadcast_message, PreparedBeliefPropagation
from ._elimination import _eliminate, VariableEliminationPlan
from ._model import (
    EnumeratedFactorGroup,
    factor_graph_log_score,
    pack_evidence,
    VariableStateValues,
)


class SmoothDualLP(StrictModule):
    """Gradient/subgradient minimization of the local-polytope dual MAP bound."""

    num_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_steps: int = 1000,
        learning_rate: float = 0.05,
        temperature: float = 0.1,
    ):
        steps = int(num_steps)
        rate = float(learning_rate)
        temp = float(temperature)
        if steps < 1:
            raise ValueError("num_steps must be positive.")
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isfinite(temp) or temp < 0.0:
            raise ValueError("temperature must be finite and non-negative.")
        self.num_steps = steps
        self.learning_rate = rate
        self.temperature = temp
        self.method_id = "smooth-dual-lp-map"


class DualLPResult(StrictModule):
    """Relaxed upper bound, decoded discrete lower bound, gap, and messages."""

    upper_bound: Array
    lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    assignment: Array
    messages: Array
    objective_history: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class PerturbAndMAPResult(StrictModule):
    """Unary-perturbation log-normalizer estimate with Monte Carlo uncertainty."""

    estimates: Array
    mean: Array
    standard_error: Array
    sample_count: int = eqx.field(static=True)
    upper_bound_in_expectation: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _smooth_max(values: Array, temperature: float, /, *, axis=None) -> Array:
    if temperature == 0.0:
        return jnp.max(values, axis=axis)
    return temperature * jsp.special.logsumexp(values / temperature, axis=axis)


def _dual_objective(
    prepared: PreparedBeliefPropagation,
    evidence: Array,
    messages: Array,
    temperature: float,
    /,
) -> Array:
    graph = prepared.graph
    indices = prepared.message_variable_state_indices
    variable_scores = evidence + segment_sum(
        messages,
        indices,
        int(prepared.state_variable_indices.shape[0]),
    )
    offsets = np.asarray(graph.variable_state_offsets)
    objective = jnp.asarray(0.0, dtype=variable_scores.dtype)
    for variable in range(graph.num_variables):
        objective = objective + _smooth_max(
            variable_scores[offsets[variable] : offsets[variable + 1]],
            temperature,
        )
    for group_index, (table, layout) in enumerate(
        zip(prepared.factor_tables, prepared.message_layout)
    ):
        group = graph.factor_groups[group_index]
        if isinstance(group, EnumeratedFactorGroup):
            joint = table
            for position, (start, stop, count, cardinality) in enumerate(layout):
                values = messages[start:stop].reshape((count, cardinality))
                indices = jnp.broadcast_to(
                    group.configurations[:, position][None, :],
                    (count, int(group.configurations.shape[0])),
                )
                joint = joint - jnp.take_along_axis(values, indices, axis=-1)
            objective = objective + jnp.sum(_smooth_max(joint, temperature, axis=-1))
            continue
        arity = len(layout)
        joint = table
        for position, (start, stop, count, cardinality) in enumerate(layout):
            values = messages[start:stop].reshape((count, cardinality))
            joint = joint - _broadcast_message(values, position, arity)
        axes = tuple(range(1, joint.ndim))
        objective = objective + jnp.sum(_smooth_max(joint, temperature, axis=axes))
    return objective


def solve_smooth_dual_lp(
    prepared: PreparedBeliefPropagation,
    method: SmoothDualLP,
    /,
    *,
    evidence: VariableStateValues | ArrayLike | None = None,
    initial_messages: ArrayLike | None = None,
) -> DualLPResult:
    """Minimize the smooth local-polytope dual and retain truthful primal/dual bounds."""
    if not isinstance(prepared, PreparedBeliefPropagation):
        raise TypeError("prepared must be PreparedBeliefPropagation.")
    if not isinstance(method, SmoothDualLP):
        raise TypeError("method must be SmoothDualLP.")
    graph = prepared.graph
    evidence_values = (
        pack_evidence(graph).values
        if evidence is None
        else evidence.values
        if isinstance(evidence, VariableStateValues)
        else pack_evidence(graph, evidence).values
    )
    messages = (
        jnp.zeros((prepared.message_count,), dtype=evidence_values.dtype)
        if initial_messages is None
        else jnp.asarray(initial_messages, dtype=evidence_values.dtype)
    )
    if messages.shape != (prepared.message_count,):
        raise ValueError("initial_messages has the wrong flat message shape.")
    objective = lambda values: _dual_objective(
        prepared, evidence_values, values, method.temperature
    )

    def step(values, _):
        current, gradient = jax.value_and_grad(objective)(values)
        updated = values - method.learning_rate * gradient
        return updated, current

    messages, history = jax.lax.scan(step, messages, xs=None, length=method.num_steps)
    upper = objective(messages)
    variable_scores = evidence_values + segment_sum(
        messages,
        prepared.message_variable_state_indices,
        int(prepared.state_variable_indices.shape[0]),
    )
    offsets = np.asarray(graph.variable_state_offsets)
    assignment = jnp.stack(
        [
            jnp.argmax(variable_scores[offsets[index] : offsets[index + 1]])
            for index in range(graph.num_variables)
        ]
    ).astype(jnp.int32)
    evidence_indices = graph.variable_state_offsets[:-1] + assignment
    lower = factor_graph_log_score(graph, assignment) + jnp.sum(
        evidence_values[evidence_indices]
    )
    gap = upper - lower
    relative = gap / jnp.maximum(1.0, jnp.maximum(jnp.abs(upper), jnp.abs(lower)))
    plan_id = canonical_fingerprint(
        {
            "kind": "smooth-dual-lp",
            "bp_plan": prepared.plan_id,
            "steps": method.num_steps,
            "learning_rate": method.learning_rate,
            "temperature": method.temperature,
        }
    )
    valid = jnp.isfinite(upper) & ~jnp.isnan(lower) & (gap >= -1e-8)
    return DualLPResult(
        upper_bound=upper,
        lower_bound=lower,
        absolute_gap=gap,
        relative_gap=relative,
        assignment=assignment,
        messages=messages,
        objective_history=history,
        valid=valid,
        plan_id=plan_id,
        method_id=method.method_id,
    )


def perturb_and_map_log_normalizer(
    plan: VariableEliminationPlan,
    /,
    *,
    key: Key[Array, ""],
    num_samples: int,
    evidence: ArrayLike | None = None,
) -> PerturbAndMAPResult:
    """Estimate an upper bound in expectation; report finite-sample Monte Carlo error."""
    if not isinstance(plan, VariableEliminationPlan):
        raise TypeError("plan must be VariableEliminationPlan.")
    count = int(num_samples)
    if count < 2:
        raise ValueError("num_samples must be at least two.")
    graph = plan.graph
    base = (
        pack_evidence(graph, evidence).values
        if evidence is not None
        else pack_evidence(graph).values
    )
    keys = jr.split(key, count)
    euler_gamma = jnp.asarray(0.5772156649015329, dtype=base.dtype)

    def one(sample_key):
        uniform = jr.uniform(
            sample_key,
            (graph.num_variable_states,),
            minval=jnp.finfo(base.dtype).eps,
            maxval=1.0,
            dtype=base.dtype,
        )
        gumbel = -jnp.log(-jnp.log(uniform))
        maximum = _eliminate(plan, base + gumbel, mode="max")
        return maximum - graph.num_variables * euler_gamma

    estimates = jax.vmap(one)(keys)
    mean = jnp.mean(estimates)
    standard_error = jnp.std(estimates, ddof=1) / jnp.sqrt(count)
    estimate_id = canonical_fingerprint(
        {
            "kind": "perturb-and-map",
            "plan": plan.plan_id,
            "samples": count,
        }
    )
    return PerturbAndMAPResult(
        estimates=estimates,
        mean=mean,
        standard_error=standard_error,
        sample_count=count,
        upper_bound_in_expectation=True,
        plan_id=estimate_id,
    )


__all__ = [
    "DualLPResult",
    "PerturbAndMAPResult",
    "SmoothDualLP",
    "perturb_and_map_log_normalizer",
    "solve_smooth_dual_lp",
]
