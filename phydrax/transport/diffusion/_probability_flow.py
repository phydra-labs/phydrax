#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
from jax import Array

from phydrax._strict import StrictModule

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._score_field import StateTimeScoreField
from ...domain import DomainFunction
from ...dynamics import ContinuousSystem, StateLayout
from ...stochastic._gaussian_diffusion import AbstractGaussianDiffusion
from ...typing import parse, PRNGKey
from ._guidance import ScoreContext


ScoreInputs: TypeAlias = Literal["bound", "args"]


def bound_score_context(
    score: StateTimeScoreField, context: ScoreContext | None, /
) -> ScoreContext:
    """Admit a score context whose names are exactly the score's declared labels."""
    resolved = ScoreContext({}) if context is None else context
    if not isinstance(resolved, ScoreContext):
        raise TypeError("context must be a ScoreContext or None.")
    if sorted(resolved.values) != sorted(score.context_labels):
        raise ValueError(
            "Score context names must be exactly the declared context labels "
            f"{score.context_labels}; got {tuple(resolved.values)}."
        )
    return resolved


def score_context_arguments(
    score: StateTimeScoreField, context: ScoreContext, /
) -> dict[str, Array] | None:
    """Context values of the score-field dependencies (``None`` without any)."""
    required = tuple(
        label
        for label in score.function.deps
        if label not in (score.state_label, score.time_label)
    )
    if not required:
        return None
    return {label: context.values[label] for label in required}


class ProbabilityFlowVectorField(StrictModule):
    """Reverse-coordinate probability-flow field induced by one learned score.

    A bound field evaluates the score with its fixed ``score_key``.  An argument
    field (``score_key is None``) reads ``(score_key, ScoreContext)`` from the
    evolution arguments, so one prepared system serves independently keyed and
    conditioned samples.
    """

    process: AbstractGaussianDiffusion
    score: StateTimeScoreField
    score_key: Array | None
    score_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def _score_inputs(self, args: Any, /) -> tuple[Array, dict[str, Array] | None]:
        if self.score_key is not None:
            return self.score_key, None
        if not (
            isinstance(args, tuple)
            and len(args) == 2
            and isinstance(args[1], ScoreContext)
        ):
            raise ValueError(
                "This probability-flow system reads (score_key, ScoreContext) "
                "through its evolution args."
            )
        score_key, context = args
        resolved = bound_score_context(self.score, context)
        return score_key, score_context_arguments(self.score, resolved)

    def __call__(self, reverse_time: Array, state: Array, args: Any, /) -> Array:
        score_key, context = self._score_inputs(args)
        forward_time = self.process.terminal_time - reverse_time
        scale = self.process.diffusion_scale(forward_time).astype(state.dtype)
        score = self.score(state, forward_time, key=score_key, context=context)
        return -self.process.drift(forward_time, state) + 0.5 * scale**2 * score


def probability_flow_system(
    process: AbstractGaussianDiffusion,
    score: DomainFunction,
    /,
    *,
    state_layout: StateLayout,
    score_id: str,
    score_key: PRNGKey | None = None,
    context_labels: Sequence[str] = (),
    score_inputs: ScoreInputs = "bound",
    state_label: str = "x",
    time_label: str = "t",
    system_id: str | None = None,
) -> ContinuousSystem:
    """Build the reverse-coordinate deterministic system for probability flow.

    ``score_inputs="bound"`` fixes one score key (``score_key``, default
    `DOC_KEY0`) and admits no context.  ``score_inputs="args"`` takes
    ``(score_key, ScoreContext)`` from the evolution arguments; the context
    names must equal ``context_labels`` and ``score_key`` must be omitted.
    """
    if not isinstance(process, AbstractGaussianDiffusion):
        raise TypeError("process must implement AbstractGaussianDiffusion.")
    if not isinstance(state_layout, StateLayout):
        raise TypeError("state_layout must be a StateLayout.")
    if state_layout.shape != process.state_shape:
        raise ValueError("state_layout shape must match the diffusion process.")
    if not state_layout.geometry.trivial:
        raise ValueError(
            "Probability flow initially requires trivial Euclidean geometry."
        )
    if not isinstance(score_id, str) or not score_id:
        raise ValueError("score_id must be a non-empty string.")
    inputs = parse(score_inputs, ScoreInputs, "score_inputs")
    labels = tuple(str(label) for label in context_labels)
    if inputs == "bound":
        if labels:
            raise ValueError("A bound probability-flow system admits no score context.")
        bound_key = DOC_KEY0 if score_key is None else score_key
    else:
        if score_key is not None:
            raise ValueError(
                "An argument probability-flow system takes its score_key through args."
            )
        bound_key = None
    adapter = StateTimeScoreField(
        score,
        state_label=state_label,
        time_label=time_label,
        context_labels=labels,
    )
    resolved_id = system_id or canonical_fingerprint(
        {
            "kind": "score-probability-flow",
            "process_id": process.process_id,
            "score_id": score_id,
            "state_shape": list(process.state_shape),
            "score_inputs": inputs,
            "context_labels": list(labels),
        }
    )
    if not isinstance(resolved_id, str) or not resolved_id:
        raise ValueError("system_id must be a non-empty string or None.")
    field = ProbabilityFlowVectorField(
        process,
        adapter,
        bound_key,
        score_id,
        canonical_fingerprint(
            {
                "kind": "probability-flow-vector-field",
                "process_id": process.process_id,
                "score_id": score_id,
                "score_inputs": inputs,
                "context_labels": list(labels),
            }
        ),
    )
    return ContinuousSystem(field, state_layout=state_layout, system_id=resolved_id)


__all__ = ["ProbabilityFlowVectorField", "probability_flow_system"]
