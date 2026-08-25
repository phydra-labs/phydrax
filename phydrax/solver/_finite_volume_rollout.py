#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._finite_volume_runtime import (
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)


FiniteVolumeRetentionPolicy: TypeAlias = Literal["final", "checkpoints", "trajectory"]
FiniteVolumeRematerializationPolicy: TypeAlias = Literal["none", "step"]
RolloutLoss = Callable[[FiniteVolumeRuntimeState, Any], Array]


class FiniteVolumeRolloutResult(StrictModule):
    final_state: FiniteVolumeRuntimeState
    retained_states: Array
    retained_times: Array
    accepted: Array
    statuses: Array
    precision_evidence: PrecisionEvidenceEnvelope


class FiniteVolumeGradientReport(StrictModule):
    directional_derivative: Array
    reverse_directional_derivative: Array
    finite_difference_derivative: Array
    jvp_vjp_residual: Array
    finite_difference_residual: Array


class FiniteVolumeRolloutPlan(StrictModule, NonTrainableState):
    runtime: PreparedFiniteVolumeRuntime
    step_count: int = eqx.field(static=True)
    retention: FiniteVolumeRetentionPolicy = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    rematerialization: FiniteVolumeRematerializationPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedFiniteVolumeRuntime,
        step_count: int,
        /,
        *,
        retention: FiniteVolumeRetentionPolicy = "final",
        checkpoint_stride: int = 1,
        rematerialization: FiniteVolumeRematerializationPolicy = "none",
    ):
        steps = int(step_count)
        stride = int(checkpoint_stride)
        if not isinstance(runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("runtime must be PreparedFiniteVolumeRuntime.")
        if steps <= 0 or stride <= 0:
            raise ValueError("Rollout step count and checkpoint stride must be positive.")
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown finite-volume retention policy.")
        if rematerialization not in ("none", "step"):
            raise ValueError("Unknown finite-volume rematerialization policy.")
        self.runtime = runtime
        self.step_count = steps
        self.retention = retention
        self.checkpoint_stride = stride
        self.rematerialization = rematerialization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-volume-rollout",
                "runtime": runtime.runtime_id,
                "step_count": steps,
                "retention": retention,
                "checkpoint_stride": stride,
                "rematerialization": rematerialization,
            }
        )

    def rollout(
        self,
        initial_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumeRolloutResult:
        if not isinstance(initial_state, FiniteVolumeRuntimeState):
            raise TypeError("initial_state must be FiniteVolumeRuntimeState.")

        def step(runtime_state, _):
            result = self.runtime.advance(runtime_state, args)
            next_state = result.runtime_state
            output = (
                next_state.content_state.conservative_content,
                next_state.time,
                result.accepted,
                next_state.last_status,
            )
            return next_state, output

        step_function = jax.checkpoint(step) if self.rematerialization == "step" else step
        final, outputs = jax.lax.scan(
            step_function,
            initial_state,
            xs=None,
            length=self.step_count,
        )
        states, times, accepted, statuses = outputs
        if self.retention == "trajectory":
            retained_states = self.runtime.precision.output(states)
            retained_times = self.runtime.precision.decision(times)
        elif self.retention == "checkpoints":
            retained_states = self.runtime.precision.checkpoint(
                states[self.checkpoint_stride - 1 :: self.checkpoint_stride]
            )
            retained_times = self.runtime.precision.decision(
                times[self.checkpoint_stride - 1 :: self.checkpoint_stride]
            )
        else:
            retained_states = self.runtime.precision.output(states[-1:])
            retained_times = self.runtime.precision.decision(times[-1:])
        return FiniteVolumeRolloutResult(
            final_state=final,
            retained_states=retained_states,
            retained_times=retained_times,
            accepted=accepted,
            statuses=statuses,
            precision_evidence=self.runtime.precision.evidence(),
        )

    def gradient_report(
        self,
        loss: RolloutLoss,
        initial_state: FiniteVolumeRuntimeState,
        tangent: ArrayLike,
        args: Any = None,
        /,
        *,
        epsilon: float = 1e-5,
    ) -> FiniteVolumeGradientReport:
        if not callable(loss):
            raise TypeError("loss must be callable.")
        direction = self.runtime.precision.storage(tangent)
        initial_content = initial_state.content_state.conservative_content
        if direction.shape != initial_content.shape:
            raise ValueError("Rollout tangent must match the conservative content.")
        epsilon_ = float(epsilon)
        if not np.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("epsilon must be finite and positive.")

        def objective(content):
            content_state = initial_state.content_state.with_content(content)
            runtime = FiniteVolumeRuntimeState(
                content_state,
                initial_state.topology_journal,
                initial_state.step_size,
                accepted_step=initial_state.accepted_step,
                last_status=initial_state.last_status,
                controller_state=initial_state.controller_state,
                integrator_state=initial_state.integrator_state,
                forcing_state=initial_state.forcing_state,
                random_state=initial_state.random_state,
                output_cursor=initial_state.output_cursor,
            )
            return loss(self.rollout(runtime, args).final_state, args)

        _, directional = jax.jvp(
            objective,
            (initial_content,),
            (direction,),
        )
        gradient = jax.grad(objective)(initial_content)
        reverse = jnp.vdot(gradient, direction)
        finite_difference = (
            objective(initial_content + epsilon_ * direction)
            - objective(initial_content - epsilon_ * direction)
        ) / (2.0 * epsilon_)
        return FiniteVolumeGradientReport(
            directional_derivative=self.runtime.precision.decision(directional),
            reverse_directional_derivative=self.runtime.precision.decision(reverse),
            finite_difference_derivative=self.runtime.precision.decision(
                finite_difference
            ),
            jvp_vjp_residual=self.runtime.precision.decision(
                jnp.abs(directional - reverse)
            ),
            finite_difference_residual=self.runtime.precision.decision(
                jnp.abs(directional - finite_difference)
            ),
        )


__all__ = [
    "FiniteVolumeGradientReport",
    "FiniteVolumeRematerializationPolicy",
    "FiniteVolumeRetentionPolicy",
    "FiniteVolumeRolloutPlan",
    "FiniteVolumeRolloutResult",
]
