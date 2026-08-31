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
from .._numerics._checkpointed_scan import checkpointed_scan
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._temporal import RealizedTemporalMesh, TemporalMesh
from ._finite_volume_runtime import (
    FiniteVolumeRunStatus,
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)


FiniteVolumeRetentionPolicy: TypeAlias = Literal["final", "checkpoints", "trajectory"]
FiniteVolumeReplayMode: TypeAlias = Literal["full", "step", "block"]
RolloutLoss = Callable[[FiniteVolumeRuntimeState, Any], Array]


class FiniteVolumeReplayPolicy(StrictModule, NonTrainableState):
    """Reverse-mode storage/recomputation policy for a fixed rollout."""

    mode: FiniteVolumeReplayMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: FiniteVolumeReplayMode = "full",
        /,
        *,
        block_size: int | None = None,
    ):
        if mode not in ("full", "step", "block"):
            raise ValueError("Unknown finite-volume replay mode.")
        size = None if block_size is None else int(block_size)
        if mode == "block":
            if size is None or size <= 0:
                raise ValueError("Block replay requires a positive block_size.")
        elif size is not None:
            raise ValueError("block_size is valid only for block replay.")
        self.mode = mode
        self.block_size = size
        self.policy_id = canonical_fingerprint(
            {"kind": "finite-volume-replay", "mode": mode, "block_size": size}
        )


class FiniteVolumeRolloutResult(StrictModule):
    final_state: FiniteVolumeRuntimeState
    retained_states: Array
    retained_times: Array
    accepted: Array
    statuses: Array
    stable_step_limits: Array | None
    stability_margins: Array | None
    realized_mesh: RealizedTemporalMesh | None
    temporal_mesh_id: str | None = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)


class FiniteVolumeGradientReport(StrictModule):
    directional_derivative: Array
    reverse_directional_derivative: Array
    finite_difference_derivative: Array
    jvp_vjp_residual: Array
    finite_difference_residual: Array


def _retained(
    retention: FiniteVolumeRetentionPolicy,
    stride: int,
    final_state: FiniteVolumeRuntimeState,
    states: Array | None,
    times: Array,
    precision,
    /,
) -> tuple[Array, Array]:
    if retention == "final":
        return (
            precision.output(final_state.content_state.conservative_content[None, ...]),
            precision.decision(final_state.time[None]),
        )
    if states is None:
        raise RuntimeError("Finite-volume rollout did not retain requested states.")
    if retention == "trajectory":
        return precision.output(states), precision.decision(times)
    return (
        precision.checkpoint(states[stride - 1 :: stride]),
        precision.decision(times[stride - 1 :: stride]),
    )


class AdaptiveFiniteVolumeRolloutPlan(StrictModule, NonTrainableState):
    """Fixed-attempt adaptive rollout that records its realized accepted mesh."""

    runtime: PreparedFiniteVolumeRuntime
    attempt_count: int = eqx.field(static=True)
    retention: FiniteVolumeRetentionPolicy = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    replay: FiniteVolumeReplayPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedFiniteVolumeRuntime,
        attempt_count: int,
        /,
        *,
        retention: FiniteVolumeRetentionPolicy = "final",
        checkpoint_stride: int = 1,
        replay: FiniteVolumeReplayPolicy | None = None,
    ):
        attempts = int(attempt_count)
        stride = int(checkpoint_stride)
        if not isinstance(runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("runtime must be PreparedFiniteVolumeRuntime.")
        if attempts <= 0 or stride <= 0:
            raise ValueError("Attempt count and checkpoint stride must be positive.")
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown finite-volume retention policy.")
        replay_ = FiniteVolumeReplayPolicy() if replay is None else replay
        if not isinstance(replay_, FiniteVolumeReplayPolicy):
            raise TypeError("replay must be FiniteVolumeReplayPolicy or None.")
        self.runtime = runtime
        self.attempt_count = attempts
        self.retention = retention
        self.checkpoint_stride = stride
        self.replay = replay_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-finite-volume-rollout",
                "runtime": runtime.runtime_id,
                "attempt_count": attempts,
                "retention": retention,
                "checkpoint_stride": stride,
                "replay": replay_.policy_id,
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
        retain_states = self.retention != "final"

        def step(runtime_state, _):
            result = self.runtime.advance(runtime_state, args)
            next_state = result.runtime_state
            common = (next_state.time, result.accepted, next_state.last_status)
            output = (
                (next_state.content_state.conservative_content, *common)
                if retain_states
                else common
            )
            return next_state, output

        indices = jnp.arange(self.attempt_count, dtype=jnp.int32)
        final, outputs = checkpointed_scan(
            step,
            initial_state,
            indices,
            length=self.attempt_count,
            mode=self.replay.mode,
            block_size=self.replay.block_size,
        )
        if retain_states:
            states, times, accepted, statuses = outputs
        else:
            states = None
            times, accepted, statuses = outputs
        retained_states, retained_times = _retained(
            self.retention,
            self.checkpoint_stride,
            final,
            states,
            times,
            self.runtime.precision,
        )
        prefix = jnp.cumprod(accepted.astype(jnp.int32)).astype(bool)
        count = jnp.sum(prefix.astype(jnp.int32))
        accepted_times = jnp.where(prefix, times, initial_state.time)
        realized = RealizedTemporalMesh(
            initial_state.time,
            accepted_times,
            prefix,
            count,
            adaptive=True,
            source_plan_id=self.plan_id,
            requested_time_id=self.plan_id,
        )
        return FiniteVolumeRolloutResult(
            final_state=final,
            retained_states=retained_states,
            retained_times=retained_times,
            accepted=accepted,
            statuses=statuses,
            stable_step_limits=None,
            stability_margins=None,
            realized_mesh=realized,
            temporal_mesh_id=None,
            precision_evidence=self.runtime.precision.evidence(),
        )


class ScheduledFiniteVolumeRolloutPlan(StrictModule, NonTrainableState):
    """Exact fixed-temporal-mesh rollout for deterministic differentiation."""

    runtime: PreparedFiniteVolumeRuntime
    temporal_mesh: TemporalMesh
    retention: FiniteVolumeRetentionPolicy = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    replay: FiniteVolumeReplayPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedFiniteVolumeRuntime,
        temporal_mesh: TemporalMesh,
        /,
        *,
        retention: FiniteVolumeRetentionPolicy = "final",
        checkpoint_stride: int = 1,
        replay: FiniteVolumeReplayPolicy | None = None,
    ):
        if not isinstance(runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("runtime must be PreparedFiniteVolumeRuntime.")
        if not isinstance(temporal_mesh, TemporalMesh):
            raise TypeError("temporal_mesh must be TemporalMesh.")
        if temporal_mesh.role != "internal" or not bool(
            np.all(np.asarray(temporal_mesh.active_intervals))
        ):
            raise ValueError(
                "Scheduled finite-volume rollout requires an all-active internal mesh."
            )
        stride = int(checkpoint_stride)
        if stride <= 0:
            raise ValueError("checkpoint_stride must be positive.")
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown finite-volume retention policy.")
        replay_ = FiniteVolumeReplayPolicy() if replay is None else replay
        if not isinstance(replay_, FiniteVolumeReplayPolicy):
            raise TypeError("replay must be FiniteVolumeReplayPolicy or None.")
        self.runtime = runtime
        self.temporal_mesh = temporal_mesh
        self.retention = retention
        self.checkpoint_stride = stride
        self.replay = replay_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scheduled-finite-volume-rollout",
                "runtime": runtime.runtime_id,
                "temporal_mesh": temporal_mesh.mesh_id,
                "retention": retention,
                "checkpoint_stride": stride,
                "replay": replay_.policy_id,
            }
        )

    @classmethod
    def from_realized_mesh(
        cls,
        runtime: PreparedFiniteVolumeRuntime,
        realized: RealizedTemporalMesh,
        /,
        **kwargs: Any,
    ) -> "ScheduledFiniteVolumeRolloutPlan":
        if not isinstance(realized, RealizedTemporalMesh):
            raise TypeError("realized must be RealizedTemporalMesh.")
        count = int(np.asarray(realized.count))
        if count <= 0:
            raise ValueError("A scheduled replay requires at least one accepted step.")
        nodes = np.concatenate(
            (
                np.asarray(realized.initial_time).reshape((1,)),
                np.asarray(realized.accepted_times)[:count],
            )
        )
        mesh = TemporalMesh(
            nodes,
            role="internal",
            realized=True,
            source_plan_id=realized.mesh_id,
        )
        return cls(runtime, mesh, **kwargs)

    def rollout(
        self,
        initial_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumeRolloutResult:
        if not isinstance(initial_state, FiniteVolumeRuntimeState):
            raise TypeError("initial_state must be FiniteVolumeRuntimeState.")
        initial_time = jnp.asarray(initial_state.time)
        tolerance = (
            32.0
            * jnp.finfo(initial_time.dtype).eps
            * jnp.maximum(jnp.abs(initial_time), 1.0)
        )
        initial_time = eqx.error_if(
            initial_time,
            jnp.abs(initial_time - self.temporal_mesh.t0) > tolerance,
            "Scheduled rollout initial time must equal temporal_mesh.t0.",
        )
        del initial_time
        retain_states = self.retention != "final"

        def step(carry, interval):
            runtime_state, active = carry
            start, step_size = interval

            def execute(_):
                current = eqx.error_if(
                    runtime_state.time,
                    jnp.abs(runtime_state.time - start) > tolerance,
                    "Scheduled rollout state time departed from its temporal mesh.",
                )
                del current
                result = self.runtime.advance_prescribed(runtime_state, step_size, args)
                next_active = active & result.accepted
                common = (
                    result.runtime_state.time,
                    result.accepted,
                    result.runtime_state.last_status,
                    result.stable_step_size,
                    result.stability_margin,
                )
                output = (
                    (result.runtime_state.content_state.conservative_content, *common)
                    if retain_states
                    else common
                )
                return (result.runtime_state, next_active), output

            def skip(_):
                common = (
                    runtime_state.time,
                    jnp.asarray(False),
                    jnp.asarray(
                        int(FiniteVolumeRunStatus.PRESCRIBED_STEP_REJECTED),
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(jnp.nan, dtype=step_size.dtype),
                    jnp.asarray(jnp.nan, dtype=step_size.dtype),
                )
                output = (
                    (runtime_state.content_state.conservative_content, *common)
                    if retain_states
                    else common
                )
                return (runtime_state, active), output

            return jax.lax.cond(active, execute, skip, operand=None)

        starts = self.temporal_mesh.nodes[:-1]
        widths = self.temporal_mesh.widths
        (final, _), outputs = checkpointed_scan(
            step,
            (initial_state, jnp.asarray(True)),
            (starts, widths),
            length=self.temporal_mesh.interval_count,
            mode=self.replay.mode,
            block_size=self.replay.block_size,
        )
        if retain_states:
            states, times, accepted, statuses, stable, margins = outputs
        else:
            states = None
            times, accepted, statuses, stable, margins = outputs
        retained_states, retained_times = _retained(
            self.retention,
            self.checkpoint_stride,
            final,
            states,
            times,
            self.runtime.precision,
        )
        return FiniteVolumeRolloutResult(
            final_state=final,
            retained_states=retained_states,
            retained_times=retained_times,
            accepted=accepted,
            statuses=statuses,
            stable_step_limits=stable,
            stability_margins=margins,
            realized_mesh=None,
            temporal_mesh_id=self.temporal_mesh.mesh_id,
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
                output_cursor=initial_state.output_cursor,
                sliding_coupling=initial_state.sliding_coupling,
                sliding_shift=initial_state.sliding_shift,
                sliding_event_id=initial_state.sliding_event_id,
            )
            return loss(self.rollout(runtime, args).final_state, args)

        _, directional = jax.jvp(objective, (initial_content,), (direction,))
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
    "AdaptiveFiniteVolumeRolloutPlan",
    "FiniteVolumeGradientReport",
    "FiniteVolumeReplayMode",
    "FiniteVolumeReplayPolicy",
    "FiniteVolumeRetentionPolicy",
    "FiniteVolumeRolloutResult",
    "ScheduledFiniteVolumeRolloutPlan",
]
