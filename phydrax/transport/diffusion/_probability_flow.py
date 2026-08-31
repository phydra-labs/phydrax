#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._score_field import StateTimeScoreField
from ...domain import DomainFunction
from ...dynamics import ContinuousSystem, StateLayout
from ...stochastic._gaussian_diffusion import AbstractGaussianDiffusion


class ProbabilityFlowVectorField(eqx.Module):
    """Reverse-coordinate probability-flow field induced by one learned score."""

    process: AbstractGaussianDiffusion
    score: StateTimeScoreField
    score_key: Array
    score_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __call__(self, reverse_time: Array, state: Array, args: Any, /) -> Array:
        del args
        forward_time = self.process.terminal_time - reverse_time
        scale = self.process.diffusion_scale(forward_time).astype(state.dtype)
        score = self.score(state, forward_time, key=self.score_key)
        return -self.process.drift(forward_time, state) + 0.5 * scale**2 * score


def probability_flow_system(
    process: AbstractGaussianDiffusion,
    score: DomainFunction,
    /,
    *,
    state_layout: StateLayout,
    score_id: str,
    score_key: Key[Array, ""] = DOC_KEY0,
    state_label: str = "x",
    time_label: str = "t",
    system_id: str | None = None,
) -> ContinuousSystem:
    """Build the reverse-coordinate deterministic system for probability flow."""
    if not isinstance(process, AbstractGaussianDiffusion):
        raise TypeError("process must implement AbstractGaussianDiffusion.")
    if not isinstance(state_layout, StateLayout):
        raise TypeError("state_layout must be a StateLayout.")
    if state_layout.shape != process.state_shape:
        raise ValueError("state_layout shape must match the diffusion process.")
    if not state_layout.geometry.trivial:
        raise ValueError("Probability flow initially requires trivial Euclidean geometry.")
    if not isinstance(score_id, str) or not score_id:
        raise ValueError("score_id must be a non-empty string.")
    adapter = StateTimeScoreField(
        score,
        state_label=state_label,
        time_label=time_label,
    )
    resolved_id = system_id or canonical_fingerprint(
        {
            "kind": "score-probability-flow",
            "process_id": process.process_id,
            "score_id": score_id,
            "state_shape": list(process.state_shape),
        }
    )
    if not isinstance(resolved_id, str) or not resolved_id:
        raise ValueError("system_id must be a non-empty string or None.")
    field = ProbabilityFlowVectorField(
        process,
        adapter,
        score_key,
        score_id,
        canonical_fingerprint(
            {
                "kind": "probability-flow-vector-field",
                "process_id": process.process_id,
                "score_id": score_id,
            }
        ),
    )
    return ContinuousSystem(field, state_layout=state_layout, system_id=resolved_id)


__all__ = ["ProbabilityFlowVectorField", "probability_flow_system"]
