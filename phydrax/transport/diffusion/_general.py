#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._score_field import StateTimeScoreField
from ..._strict import StrictModule
from ...dynamics import ContinuousSystem, StateLayout
from ...stochastic._general_diffusion import AbstractItoScoreDiffusion
from ._guidance import GuidedScoreField, ScoreContext


def _evaluate_score(score, state, time, context, key, /):
    if isinstance(score, GuidedScoreField):
        return score(state, time, context=context, key=key)
    required = tuple(
        dependency
        for dependency in score.function.deps
        if dependency not in (score.state_label, score.time_label)
    )
    if any(name not in context.values for name in required):
        raise ValueError("Score context is missing a score-field dependency.")
    values = {name: context.values[name] for name in required}
    return score(state, time, context=values, key=key)


class _GeneralReverseDrift(eqx.Module):
    process: AbstractItoScoreDiffusion
    score: Any

    def __call__(self, reverse_time, state, args, /):
        score_key, context = args
        score = _evaluate_score(
            self.score,
            state,
            self.process.terminal_time - reverse_time,
            context,
            score_key,
        )
        return self.process.reverse_drift(reverse_time, state, score)


class _GeneralProbabilityFlowField(eqx.Module):
    process: AbstractItoScoreDiffusion
    score: Any
    score_key: Array
    context: ScoreContext

    def __call__(self, reverse_time, state, args, /):
        del args
        forward_time = self.process.terminal_time - reverse_time
        score = _evaluate_score(
            self.score,
            state,
            forward_time,
            self.context,
            self.score_key,
        )
        return self.process.probability_flow_drift(reverse_time, state, score)


class _GeneralDiffusionOperator(eqx.Module):
    process: AbstractItoScoreDiffusion

    def __call__(self, reverse_time, state, args, /):
        del args
        forward_time = self.process.terminal_time - reverse_time
        factor = self.process.diffusion_factor(forward_time, state)
        return lx.MatrixLinearOperator(factor)


class GeneralReverseProblem(StrictModule):
    """Canonical differential problem for matrix/state-dependent reverse diffusion."""

    problem: Any
    context: ScoreContext
    score_key: Array
    process_id: str = eqx.field(static=True)
    score_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)


def general_reverse_diffusion_problem(
    process: AbstractItoScoreDiffusion,
    score: StateTimeScoreField | GuidedScoreField,
    initial_state: ArrayLike,
    /,
    *,
    context: ScoreContext | None = None,
    score_key: Key[Array, ""] = DOC_KEY0,
    score_id: str,
    problem_id: str | None = None,
) -> GeneralReverseProblem:
    """Build a canonical reverse-time Itô problem with operator-valued noise."""
    from ...solver._differential import DifferentialProblem, WienerTerm

    if not isinstance(process, AbstractItoScoreDiffusion):
        raise TypeError("process must implement AbstractItoScoreDiffusion.")
    if not isinstance(score, (StateTimeScoreField, GuidedScoreField)):
        raise TypeError("score must be a state-time or guided score field.")
    if not score_id:
        raise ValueError("score_id must be non-empty.")
    resolved_context = ScoreContext({}) if context is None else context
    if not isinstance(resolved_context, ScoreContext):
        raise TypeError("context must be a ScoreContext or None.")
    state = jnp.asarray(initial_state)
    if state.shape != process.state_shape:
        raise ValueError("initial_state must match the diffusion state shape.")
    probe_factor = process.diffusion_factor(process.terminal_time, state)
    noise_shape = (int(probe_factor.shape[-1]),)
    drift = _GeneralReverseDrift(process, score)
    coefficient = _GeneralDiffusionOperator(process)
    term = WienerTerm(
        "general-reverse-diffusion",
        coefficient,
        noise_shape,
        structure="general",
        basis_id=process.process_id,
        representation="operator",
    )
    identifier = problem_id or canonical_fingerprint(
        {
            "kind": "general-reverse-diffusion-problem",
            "process_id": process.process_id,
            "score_id": score_id,
            "context_id": resolved_context.context_id,
        }
    )
    problem = DifferentialProblem(
        drift,
        state,
        t0=0.0,
        t1=process.terminal_time,
        args=(jnp.asarray(score_key), resolved_context),
        wiener_terms=(term,),
        interpretation="ito",
        problem_id=identifier,
    )
    return GeneralReverseProblem(
        problem,
        resolved_context,
        jnp.asarray(score_key),
        process.process_id,
        score_id,
        identifier,
    )


def general_probability_flow_system(
    process: AbstractItoScoreDiffusion,
    score: StateTimeScoreField | GuidedScoreField,
    /,
    *,
    context: ScoreContext | None = None,
    score_key: Key[Array, ""] = DOC_KEY0,
    score_id: str,
    state_layout: StateLayout,
    system_id: str | None = None,
) -> ContinuousSystem:
    """Build a deterministic probability-flow system for a general Itô process."""
    if not isinstance(process, AbstractItoScoreDiffusion):
        raise TypeError("process must implement AbstractItoScoreDiffusion.")
    if not isinstance(score, (StateTimeScoreField, GuidedScoreField)):
        raise TypeError("score must be a state-time or guided score field.")
    if not score_id:
        raise ValueError("score_id must be non-empty.")
    if state_layout.shape != process.state_shape or not state_layout.geometry.trivial:
        raise ValueError("General probability flow requires matching trivial state layout.")
    resolved_context = ScoreContext({}) if context is None else context
    if not isinstance(resolved_context, ScoreContext):
        raise TypeError("context must be a ScoreContext or None.")
    identifier = system_id or canonical_fingerprint(
        {
            "kind": "general-probability-flow",
            "process_id": process.process_id,
            "score_id": score_id,
            "context_id": resolved_context.context_id,
        }
    )
    field = _GeneralProbabilityFlowField(
        process,
        score,
        jnp.asarray(score_key),
        resolved_context,
    )
    return ContinuousSystem(field, state_layout=state_layout, system_id=identifier)


__all__ = [
    "GeneralReverseProblem",
    "general_probability_flow_system",
    "general_reverse_diffusion_problem",
]
