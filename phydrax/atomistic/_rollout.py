#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._numerics._checkpointed_scan import checkpointed_scan
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics


AtomisticReplayMode: TypeAlias = Literal["full", "step", "block"]
AtomisticRetention: TypeAlias = Literal["final", "trajectory"]


class AtomisticReplayPolicy(StrictModule, NonTrainableState):
    mode: AtomisticReplayMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: AtomisticReplayMode = "full",
        /,
        *,
        block_size: int | None = None,
    ):
        if mode not in ("full", "step", "block"):
            raise ValueError("Unknown atomistic replay mode.")
        size = None if block_size is None else int(block_size)
        if mode == "block":
            if size is None or size <= 0:
                raise ValueError("Block replay requires a positive block_size.")
        elif size is not None:
            raise ValueError("block_size is valid only for block replay.")
        self.mode = mode
        self.block_size = size
        self.policy_id = canonical_fingerprint(
            {"kind": "atomistic-replay-policy", "mode": mode, "block_size": size}
        )


class AtomisticTrajectoryPlan(StrictModule, NonTrainableState):
    step_count: int = eqx.field(static=True)
    sample_stride: int = eqx.field(static=True)
    include_initial: bool = eqx.field(static=True)
    retention: AtomisticRetention = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_count: int,
        /,
        *,
        sample_stride: int = 1,
        include_initial: bool = True,
        retention: AtomisticRetention = "trajectory",
    ):
        steps = int(step_count)
        stride = int(sample_stride)
        if steps <= 0 or stride <= 0:
            raise ValueError("step_count and sample_stride must be positive.")
        if retention not in ("final", "trajectory"):
            raise ValueError("retention must be 'final' or 'trajectory'.")
        scheduled = steps // stride
        if steps % stride:
            scheduled += 1
        capacity = 1 if retention == "final" else scheduled + int(include_initial)
        self.step_count = steps
        self.sample_stride = stride
        self.include_initial = bool(include_initial)
        self.retention = retention
        self.capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-trajectory-plan",
                "step_count": steps,
                "sample_stride": stride,
                "include_initial": bool(include_initial),
                "retention": retention,
                "capacity": capacity,
            }
        )


class AtomisticTrajectory(StrictModule):
    times: Array
    positions: Array
    momenta: Array
    image_counts: Array
    cells: Array
    energies: Array
    valid: Array
    count: Array
    trajectory_id: str = eqx.field(static=True)

    @property
    def sample_mask(self) -> Array:
        return jnp.arange(self.times.shape[0], dtype=jnp.int32) < self.count


class AtomisticReplayRecord(StrictModule):
    accepted_steps: Array
    rejected_steps: Array
    route_digest: Array
    image_digest: Array
    stochastic_digest: Array
    successful: Array
    replay_id: str = eqx.field(static=True)


class AtomisticRolloutResult(StrictModule):
    final_state: AtomisticDynamicsState
    trajectory: AtomisticTrajectory
    replay: AtomisticReplayRecord
    successful: Array
    rollout_id: str = eqx.field(static=True)


class AtomisticRolloutPlan(StrictModule):
    dynamics: PreparedAtomisticDynamics
    trajectory: AtomisticTrajectoryPlan
    replay: AtomisticReplayPolicy
    rollout_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        trajectory: AtomisticTrajectoryPlan,
        /,
        *,
        replay: AtomisticReplayPolicy | None = None,
    ):
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(trajectory, AtomisticTrajectoryPlan):
            raise TypeError("trajectory must be AtomisticTrajectoryPlan.")
        replay_ = AtomisticReplayPolicy() if replay is None else replay
        if not isinstance(replay_, AtomisticReplayPolicy):
            raise TypeError("replay must be AtomisticReplayPolicy or None.")
        self.dynamics = dynamics
        self.trajectory = trajectory
        self.replay = replay_
        self.rollout_id = canonical_fingerprint(
            {
                "kind": "atomistic-rollout-plan",
                "dynamics": dynamics.prepared_id,
                "trajectory": trajectory.plan_id,
                "replay": replay_.policy_id,
            }
        )

    def rollout(self, initial_state: AtomisticDynamicsState, /) -> AtomisticRolloutResult:
        if not isinstance(initial_state, AtomisticDynamicsState):
            raise TypeError("initial_state must be AtomisticDynamicsState.")
        if initial_state.prepared_dynamics_id != self.dynamics.prepared_id:
            raise ValueError("Initial state belongs to another dynamics runtime.")
        state = initial_state
        dtype = state.kinematics.positions.dtype
        capacity = self.trajectory.capacity
        particle_capacity = self.dynamics.system.capacity
        cell_shape = state.cell_vectors.shape
        times = jnp.full((capacity,), jnp.inf, dtype=dtype)
        positions = jnp.zeros((capacity, particle_capacity, 3), dtype=dtype)
        momenta = jnp.zeros_like(positions)
        image_counts = jnp.zeros((capacity, particle_capacity, 3), dtype=jnp.int32)
        cells = jnp.zeros((capacity,) + cell_shape, dtype=dtype)
        energies = jnp.zeros((capacity, 3), dtype=dtype)
        valid = jnp.zeros((capacity,), dtype=bool)
        initial_count = jnp.asarray(
            int(
                self.trajectory.retention == "trajectory"
                and self.trajectory.include_initial
            ),
            dtype=jnp.int32,
        )
        if self.trajectory.retention == "trajectory" and self.trajectory.include_initial:
            times = times.at[0].set(state.time)
            positions = positions.at[0].set(state.kinematics.positions)
            momenta = momenta.at[0].set(state.kinematics.momenta)
            image_counts = image_counts.at[0].set(state.kinematics.image_counts)
            cells = cells.at[0].set(state.cell_vectors)
            energies = energies.at[0].set(
                jnp.stack(
                    (
                        state.energy.kinetic_energy,
                        state.energy.potential_energy,
                        state.energy.total_energy,
                    )
                )
            )
            valid = valid.at[0].set(True)

        initial_carry = (
            state,
            jnp.asarray(True),
            times,
            positions,
            momenta,
            image_counts,
            cells,
            energies,
            valid,
            initial_count,
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.uint64),
            jnp.zeros((), dtype=jnp.uint64),
            jnp.zeros((), dtype=jnp.uint64),
        )

        def advance(carry, index):
            (
                current,
                cumulative_success,
                time_buffer,
                position_buffer,
                momentum_buffer,
                image_buffer,
                cell_buffer,
                energy_buffer,
                valid_buffer,
                count,
                accepted_count,
                rejected_count,
                route_digest,
                image_digest,
                stochastic_digest,
            ) = carry
            result = self.dynamics.step_detailed(current)
            step_success = cumulative_success & result.successful
            next_state = tree_where(cumulative_success, result.accepted_state, current)
            cumulative = cumulative_success & result.successful
            last = index + 1 == self.trajectory.step_count
            requested = self.trajectory.retention == "trajectory" and (
                ((index + 1) % self.trajectory.sample_stride == 0) | last
            )
            write = jnp.asarray(requested) & (count < capacity)
            destination = jnp.minimum(count, capacity - 1)
            time_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(next_state.time),
                lambda value: value,
                time_buffer,
            )
            position_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(next_state.kinematics.positions),
                lambda value: value,
                position_buffer,
            )
            momentum_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(next_state.kinematics.momenta),
                lambda value: value,
                momentum_buffer,
            )
            image_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(
                    next_state.kinematics.image_counts
                ),
                lambda value: value,
                image_buffer,
            )
            cell_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(next_state.cell_vectors),
                lambda value: value,
                cell_buffer,
            )
            energy_value = jnp.stack(
                (
                    next_state.energy.kinetic_energy,
                    next_state.energy.potential_energy,
                    next_state.energy.total_energy,
                )
            )
            energy_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(energy_value),
                lambda value: value,
                energy_buffer,
            )
            valid_buffer = jax.lax.cond(
                write,
                lambda value: value.at[destination].set(step_success),
                lambda value: value,
                valid_buffer,
            )
            cache_epoch = (
                jnp.zeros((), dtype=jnp.int32)
                if next_state.neighborhood_cache is None
                else next_state.neighborhood_cache.epoch
            )
            route_digest = route_digest * jnp.uint64(1099511628211) + jnp.asarray(
                cache_epoch.astype(jnp.uint32), dtype=jnp.uint64
            )
            route_digest = route_digest + jnp.asarray(
                result.rejection_reasons.astype(jnp.uint32), dtype=jnp.uint64
            )
            image_digest = image_digest * jnp.uint64(1099511628211) + jnp.asarray(
                jnp.sum(next_state.kinematics.image_counts, dtype=jnp.int64),
                dtype=jnp.uint64,
            )
            stochastic_digest = stochastic_digest * jnp.uint64(
                1099511628211
            ) + jnp.asarray(next_state.step_index.astype(jnp.uint32), dtype=jnp.uint64)
            return (
                next_state,
                cumulative,
                time_buffer,
                position_buffer,
                momentum_buffer,
                image_buffer,
                cell_buffer,
                energy_buffer,
                valid_buffer,
                count + write.astype(jnp.int32),
                accepted_count
                + (cumulative_success & result.successful).astype(jnp.int32),
                rejected_count
                + (cumulative_success & ~result.successful).astype(jnp.int32),
                route_digest,
                image_digest,
                stochastic_digest,
            ), None

        indices = jnp.arange(self.trajectory.step_count, dtype=jnp.int32)
        final, _ = checkpointed_scan(
            advance,
            initial_carry,
            indices,
            length=self.trajectory.step_count,
            mode=self.replay.mode,
            block_size=self.replay.block_size,
        )
        final_state = final[0]
        successful = final[1]
        if self.trajectory.retention == "final":
            times = final[2].at[0].set(final_state.time)
            positions = final[3].at[0].set(final_state.kinematics.positions)
            momenta = final[4].at[0].set(final_state.kinematics.momenta)
            image_counts = final[5].at[0].set(final_state.kinematics.image_counts)
            cells = final[6].at[0].set(final_state.cell_vectors)
            energies = (
                final[7]
                .at[0]
                .set(
                    jnp.stack(
                        (
                            final_state.energy.kinetic_energy,
                            final_state.energy.potential_energy,
                            final_state.energy.total_energy,
                        )
                    )
                )
            )
            valid = final[8].at[0].set(successful)
            count = jnp.asarray(1, dtype=jnp.int32)
        else:
            times, positions, momenta, image_counts, cells, energies, valid, count = (
                final[2:10]
            )
        trajectory = AtomisticTrajectory(
            times=times,
            positions=positions,
            momenta=momenta,
            image_counts=image_counts,
            cells=cells,
            energies=energies,
            valid=valid,
            count=count,
            trajectory_id=canonical_fingerprint(
                {"kind": "atomistic-trajectory", "rollout": self.rollout_id}
            ),
        )
        replay = AtomisticReplayRecord(
            accepted_steps=final[10],
            rejected_steps=final[11],
            route_digest=final[12],
            image_digest=final[13],
            stochastic_digest=final[14],
            successful=successful,
            replay_id=canonical_fingerprint(
                {"kind": "atomistic-replay", "rollout": self.rollout_id}
            ),
        )
        return AtomisticRolloutResult(
            final_state=final_state,
            trajectory=trajectory,
            replay=replay,
            successful=successful,
            rollout_id=self.rollout_id,
        )


def atomistic_replay_matches(
    left: AtomisticReplayRecord, right: AtomisticReplayRecord, /
) -> Array:
    if not isinstance(left, AtomisticReplayRecord) or not isinstance(
        right, AtomisticReplayRecord
    ):
        raise TypeError("Both values must be AtomisticReplayRecord instances.")
    return (
        (left.accepted_steps == right.accepted_steps)
        & (left.rejected_steps == right.rejected_steps)
        & (left.route_digest == right.route_digest)
        & (left.image_digest == right.image_digest)
        & (left.stochastic_digest == right.stochastic_digest)
        & (left.successful == right.successful)
    )


__all__ = [
    "AtomisticReplayPolicy",
    "AtomisticReplayRecord",
    "AtomisticRolloutPlan",
    "AtomisticRolloutResult",
    "AtomisticTrajectory",
    "AtomisticTrajectoryPlan",
    "atomistic_replay_matches",
]
