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
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._numerics._checkpointed_scan import checkpointed_scan
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import TemporalMesh
from ..discretization.mpm import (
    MPMParticleState,
    MPMRejectionReason,
    MPMRunStatus,
    MPMRuntimeState,
    PreparedMPMDynamics,
)
from ..equations import MaterialPointArguments


MPMReplayMode: TypeAlias = Literal["full", "step", "block"]
MPMRetentionMode: TypeAlias = Literal["final", "checkpoints", "trajectory"]
MPMGradientKind: TypeAlias = Literal["piecewise-discrete", "frozen-surrogate"]
MPMRolloutLoss = Callable[[MPMRuntimeState, MaterialPointArguments], Array]


class MPMReplayPolicy(StrictModule, NonTrainableState):
    mode: MPMReplayMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: MPMReplayMode = "full",
        /,
        *,
        block_size: int | None = None,
    ):
        if mode not in ("full", "step", "block"):
            raise ValueError("Unknown MPM replay mode.")
        size = None if block_size is None else int(block_size)
        if mode == "block":
            if size is None or size <= 0:
                raise ValueError("Block MPM replay requires a positive block_size.")
        elif size is not None:
            raise ValueError("block_size is valid only for block MPM replay.")
        self.mode = mode
        self.block_size = size
        self.policy_id = canonical_fingerprint(
            {"kind": "mpm-replay", "mode": mode, "block_size": size}
        )


class MPMRetainedTrajectory(StrictModule):
    particles: MPMParticleState
    times: Array


class MPMReplayEvidence(StrictModule, NonTrainableState):
    replay_policy_id: str = eqx.field(static=True)
    accumulation_policy_id: str = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    retained_state_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MPMRolloutResult(StrictModule):
    final_state: MPMRuntimeState
    retained: MPMRetainedTrajectory
    accepted: Array
    statuses: Array
    rejection_reasons: Array
    stable_step_limits: Array
    stability_margins: Array
    route_digests: Array
    transfer_successful: Array
    relative_mass_defects: Array
    relative_momentum_defects: Array
    relative_angular_momentum_defects: Array
    energy_balance_defects: Array
    minimum_jacobians: Array
    maximum_apic_conditions: Array
    replay_evidence: MPMReplayEvidence = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    temporal_mesh_id: str = eqx.field(static=True)


class MPMGradientReport(StrictModule):
    directional_derivative: Array
    reverse_directional_derivative: Array
    finite_difference_derivative: Array
    jvp_vjp_residual: Array
    finite_difference_residual: Array
    branch_matched: Array
    gradient_kind: MPMGradientKind = eqx.field(static=True)


def _tree_vdot(left: Any, right: Any, /) -> Array:
    terms = []
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        if eqx.is_inexact_array(left_leaf) and eqx.is_inexact_array(right_leaf):
            terms.append(jnp.vdot(left_leaf, right_leaf))
    if not terms:
        return jnp.asarray(0.0)
    return sum(terms[1:], terms[0])


def _tree_perturb(value: Any, direction: Any, scale: float, /) -> Any:
    return jax.tree.map(
        lambda primal, tangent: (
            primal + scale * tangent
            if eqx.is_inexact_array(primal) and eqx.is_inexact_array(tangent)
            else primal
        ),
        value,
        direction,
        is_leaf=lambda leaf: leaf is None,
    )


def _retain_particles(
    mode: MPMRetentionMode,
    stride: int,
    final_state: MPMRuntimeState,
    states: MPMParticleState | None,
    times: Array,
    /,
) -> MPMRetainedTrajectory:
    if mode == "final":
        particles = jax.tree.map(lambda value: value[None, ...], final_state.particles)
        return MPMRetainedTrajectory(particles, final_state.time[None])
    if states is None:
        raise RuntimeError("MPM rollout did not retain requested particle states.")
    if mode == "trajectory":
        return MPMRetainedTrajectory(states, times)
    indices = jnp.arange(stride - 1, times.shape[0], stride, dtype=jnp.int32)
    particles = jax.tree.map(lambda value: value[indices], states)
    return MPMRetainedTrajectory(particles, times[indices])


class ScheduledMPMRolloutPlan(StrictModule, NonTrainableState):
    """Exact fixed-temporal material-point rollout with explicit replay policy."""

    dynamics: PreparedMPMDynamics
    temporal_mesh: TemporalMesh
    retention: MPMRetentionMode = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    replay: MPMReplayPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedMPMDynamics,
        temporal_mesh: TemporalMesh,
        /,
        *,
        retention: MPMRetentionMode = "final",
        checkpoint_stride: int = 1,
        replay: MPMReplayPolicy | None = None,
    ):
        if not isinstance(dynamics, PreparedMPMDynamics):
            raise TypeError("dynamics must be PreparedMPMDynamics.")
        if not isinstance(temporal_mesh, TemporalMesh):
            raise TypeError("temporal_mesh must be TemporalMesh.")
        if temporal_mesh.role != "internal" or not bool(
            np.all(np.asarray(temporal_mesh.active_intervals))
        ):
            raise ValueError(
                "Scheduled MPM rollout requires an all-active internal mesh."
            )
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown MPM retention mode.")
        stride = int(checkpoint_stride)
        if stride <= 0:
            raise ValueError("checkpoint_stride must be positive.")
        replay_ = MPMReplayPolicy() if replay is None else replay
        if not isinstance(replay_, MPMReplayPolicy):
            raise TypeError("replay must be MPMReplayPolicy or None.")
        accumulation = dynamics.splat.plan.execution.accumulation
        if replay_.mode in ("step", "block") and accumulation == "fast":
            raise ValueError(
                "Rematerialized MPM replay requires deterministic or compensated accumulation."
            )
        self.dynamics = dynamics
        self.temporal_mesh = temporal_mesh
        self.retention = retention
        self.checkpoint_stride = stride
        self.replay = replay_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scheduled-mpm-rollout",
                "dynamics": dynamics.prepared_id,
                "temporal_mesh": temporal_mesh.mesh_id,
                "retention": retention,
                "checkpoint_stride": stride,
                "replay": replay_.policy_id,
            }
        )

    def rollout(
        self,
        initial_state: MPMRuntimeState,
        arguments: MaterialPointArguments,
        /,
    ) -> MPMRolloutResult:
        if not isinstance(initial_state, MPMRuntimeState):
            raise TypeError("initial_state must be MPMRuntimeState.")
        if not isinstance(arguments, MaterialPointArguments):
            raise TypeError("arguments must be MaterialPointArguments.")
        initial_time = jnp.asarray(initial_state.time)
        tolerance = (
            32.0
            * jnp.finfo(initial_time.dtype).eps
            * jnp.maximum(jnp.abs(initial_time), 1.0)
        )
        initial_time = eqx.error_if(
            initial_time,
            jnp.abs(initial_time - self.temporal_mesh.t0) > tolerance,
            "Scheduled MPM rollout initial time must equal temporal_mesh.t0.",
        )
        del initial_time
        retain_states = self.retention != "final"

        def step(carry, interval):
            runtime_state, active = carry
            start, width = interval

            def execute(_):
                current = eqx.error_if(
                    runtime_state.time,
                    jnp.abs(runtime_state.time - start) > tolerance,
                    "Scheduled MPM state time departed from its temporal mesh.",
                )
                del current
                detail = self.dynamics.step_detailed(runtime_state, width, arguments)
                next_active = active & detail.successful
                common = (
                    detail.accepted_state.time,
                    detail.successful,
                    detail.accepted_state.last_status,
                    detail.rejection_reasons,
                    detail.restriction.selected,
                    detail.stability_margin,
                    detail.diagnostics.transfer.route_digest,
                    detail.diagnostics.transfer.successful,
                    detail.diagnostics.transfer.relative_mass_defect,
                    detail.diagnostics.transfer.relative_momentum_defect,
                    detail.diagnostics.transfer.relative_angular_momentum_defect,
                    detail.diagnostics.energy.balance_defect,
                    detail.diagnostics.minimum_jacobian,
                    detail.diagnostics.transfer.maximum_apic_condition,
                )
                output = (
                    (detail.accepted_state.particles, *common)
                    if retain_states
                    else common
                )
                return (detail.accepted_state, next_active), output

            def skip(_):
                dtype = runtime_state.time.dtype
                common = (
                    runtime_state.time,
                    jnp.asarray(False),
                    jnp.asarray(
                        int(MPMRunStatus.PRESCRIBED_STEP_REJECTED), dtype=jnp.int32
                    ),
                    jnp.asarray(int(MPMRejectionReason.STABILITY), dtype=jnp.int32),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.zeros((), dtype=jnp.int64),
                    jnp.asarray(False),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                )
                output = (runtime_state.particles, *common) if retain_states else common
                skipped = MPMRuntimeState(
                    runtime_state.particles,
                    runtime_state.time,
                    runtime_state.accepted_step,
                    common[2],
                )
                return (skipped, active), output

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
            (
                states,
                times,
                accepted,
                statuses,
                reasons,
                stable,
                margins,
                digests,
                transfer_successful,
                mass_defects,
                momentum_defects,
                angular_defects,
                energy_defects,
                minimum_jacobians,
                maximum_apic_conditions,
            ) = outputs
        else:
            states = None
            (
                times,
                accepted,
                statuses,
                reasons,
                stable,
                margins,
                digests,
                transfer_successful,
                mass_defects,
                momentum_defects,
                angular_defects,
                energy_defects,
                minimum_jacobians,
                maximum_apic_conditions,
            ) = outputs
        retained = _retain_particles(
            self.retention,
            self.checkpoint_stride,
            final,
            states,
            times,
        )
        accumulation = self.dynamics.splat.plan.execution.accumulation
        evidence = MPMReplayEvidence(
            self.replay.policy_id,
            self.dynamics.splat.plan.execution.policy_id,
            accumulation != "fast",
            int(retained.times.shape[0]),
            canonical_fingerprint(
                {
                    "kind": "mpm-replay-evidence",
                    "plan": self.plan_id,
                    "replay": self.replay.policy_id,
                    "accumulation": accumulation,
                    "retained_state_count": int(retained.times.shape[0]),
                }
            ),
        )
        return MPMRolloutResult(
            final,
            retained,
            accepted,
            statuses,
            reasons,
            stable,
            margins,
            digests,
            transfer_successful,
            mass_defects,
            momentum_defects,
            angular_defects,
            energy_defects,
            minimum_jacobians,
            maximum_apic_conditions,
            evidence,
            self.dynamics.precision_evidence,
            self.temporal_mesh.mesh_id,
        )

    def gradient_report(
        self,
        loss: MPMRolloutLoss,
        initial_state: MPMRuntimeState,
        arguments: MaterialPointArguments,
        state_direction: MPMParticleState,
        argument_direction: MaterialPointArguments,
        /,
        *,
        epsilon: float = 1.0e-5,
    ) -> MPMGradientReport:
        if not callable(loss):
            raise TypeError("loss must be callable.")
        epsilon_ = float(epsilon)
        if not np.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("epsilon must be finite and positive.")

        def objective(particle_state, argument_values):
            runtime = MPMRuntimeState(
                particle_state,
                initial_state.time,
                initial_state.accepted_step,
                initial_state.last_status,
            )
            final = self.rollout(runtime, argument_values).final_state
            return jnp.asarray(loss(final, argument_values))

        primals = (initial_state.particles, arguments)
        tangents = (state_direction, argument_direction)
        _, directional = jax.jvp(objective, primals, tangents)
        gradients = jax.grad(objective, argnums=(0, 1))(*primals)
        reverse = _tree_vdot(gradients[0], state_direction) + _tree_vdot(
            gradients[1], argument_direction
        )
        kind: MPMGradientKind = (
            "piecewise-discrete"
            if self.dynamics.splat.plan.execution.geometry_ad == "piecewise"
            else "frozen-surrogate"
        )
        if kind == "piecewise-discrete":
            plus_state = _tree_perturb(initial_state.particles, state_direction, epsilon_)
            minus_state = _tree_perturb(
                initial_state.particles, state_direction, -epsilon_
            )
            plus_args = _tree_perturb(arguments, argument_direction, epsilon_)
            minus_args = _tree_perturb(arguments, argument_direction, -epsilon_)
            plus_runtime = MPMRuntimeState(
                plus_state,
                initial_state.time,
                initial_state.accepted_step,
                initial_state.last_status,
            )
            minus_runtime = MPMRuntimeState(
                minus_state,
                initial_state.time,
                initial_state.accepted_step,
                initial_state.last_status,
            )
            plus = self.rollout(plus_runtime, plus_args)
            minus = self.rollout(minus_runtime, minus_args)
            finite_difference = (
                loss(plus.final_state, plus_args) - loss(minus.final_state, minus_args)
            ) / (2.0 * epsilon_)
            branch_matched = (
                jnp.array_equal(plus.route_digests, minus.route_digests)
                & jnp.array_equal(plus.statuses, minus.statuses)
                & jnp.array_equal(plus.accepted, minus.accepted)
            )
            finite_residual = jnp.where(
                branch_matched,
                jnp.abs(directional - finite_difference),
                jnp.asarray(jnp.nan, dtype=jnp.asarray(directional).dtype),
            )
        else:
            finite_difference = jnp.asarray(jnp.nan, dtype=jnp.asarray(directional).dtype)
            finite_residual = jnp.asarray(jnp.nan, dtype=jnp.asarray(directional).dtype)
            branch_matched = jnp.asarray(False)
        return MPMGradientReport(
            directional,
            reverse,
            finite_difference,
            jnp.abs(directional - reverse),
            finite_residual,
            branch_matched,
            kind,
        )


__all__ = [
    "MPMGradientKind",
    "MPMGradientReport",
    "MPMReplayMode",
    "MPMReplayPolicy",
    "MPMReplayEvidence",
    "MPMRetentionMode",
    "MPMRetainedTrajectory",
    "MPMRolloutResult",
    "ScheduledMPMRolloutPlan",
]
