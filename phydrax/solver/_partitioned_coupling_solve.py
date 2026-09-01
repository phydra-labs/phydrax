#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isclose, isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._numerics._checkpointed_scan import checkpointed_scan
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._fixed_step import FixedStepReplayPolicy
from ._partitioned_coupling_graph import (
    CouplingGraph,
    CouplingResourcePolicy,
    prepare_coupling,
    PreparedCoupling,
)
from ._partitioned_coupling_runtime import advance_coupling_window
from ._partitioned_coupling_types import (
    AbstractCouplingPolicy,
    CouplingDifferentiationPolicy,
    CouplingState,
)


CouplingRetentionPolicy: TypeAlias = Literal["final", "checkpoints", "trajectory"]


def _prepend(initial: Any, values: Any, /) -> Any:
    return jax.tree.map(
        lambda first, rest: jnp.concatenate((first[None, ...], rest), axis=0),
        initial,
        values,
    )


def _singleton(value: Any, /) -> Any:
    return jax.tree.map(lambda leaf: leaf[None, ...], value)


class CouplingProblem(StrictModule, NonTrainableState):
    """Fixed-window partitioned coupling problem and initial accepted state."""

    graph: CouplingGraph
    participant_states: tuple[Any, ...]
    exchange_values: tuple[Any, ...]
    policy: AbstractCouplingPolicy
    differentiation: CouplingDifferentiationPolicy
    args: Any
    resources: CouplingResourcePolicy
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    window_size: float = eqx.field(static=True)
    window_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        graph: CouplingGraph,
        participant_states: tuple[Any, ...],
        exchange_values: tuple[Any, ...],
        policy: AbstractCouplingPolicy,
        /,
        *,
        t0: float,
        t1: float,
        window_size: float,
        differentiation: CouplingDifferentiationPolicy | None = None,
        args: Any = None,
        resources: CouplingResourcePolicy | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(graph, CouplingGraph):
            raise TypeError("graph must be CouplingGraph.")
        if not isinstance(policy, AbstractCouplingPolicy):
            raise TypeError("policy must be AbstractCouplingPolicy.")
        differentiation_ = (
            CouplingDifferentiationPolicy()
            if differentiation is None
            else differentiation
        )
        if not isinstance(differentiation_, CouplingDifferentiationPolicy):
            raise TypeError(
                "differentiation must be CouplingDifferentiationPolicy or None."
            )
        resources_ = CouplingResourcePolicy() if resources is None else resources
        if not isinstance(resources_, CouplingResourcePolicy):
            raise TypeError("resources must be CouplingResourcePolicy or None.")
        start = float(t0)
        end = float(t1)
        size = float(window_size)
        if not isfinite(start) or not isfinite(end) or end <= start:
            raise ValueError("Coupling problem requires finite t1 > t0.")
        if not isfinite(size) or size <= 0.0:
            raise ValueError("Coupling window_size must be finite and positive.")
        raw_count = (end - start) / size
        count = round(raw_count)
        if count <= 0 or not isclose(raw_count, count, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Coupling interval must contain an integer window count.")
        generated = canonical_fingerprint(
            {
                "kind": "coupling-problem",
                "graph": graph.graph_id,
                "policy": policy.policy_id,
                "differentiation": differentiation_.policy_id,
                "t0": start,
                "t1": end,
                "window_size": size,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("Coupling problem_id must be non-empty.")
        self.graph = graph
        self.participant_states = tuple(participant_states)
        self.exchange_values = tuple(exchange_values)
        self.policy = policy
        self.differentiation = differentiation_
        self.args = args
        self.resources = resources_
        self.t0 = start
        self.t1 = end
        self.window_size = size
        self.window_count = count
        self.problem_id = identifier

    def prepare(self, /) -> PreparedCoupling:
        return prepare_coupling(
            self.graph,
            self.participant_states,
            self.exchange_values,
            policy=self.policy,
            differentiation=self.differentiation,
            time=self.t0,
            args=self.args,
            problem_id=self.problem_id,
            resources=self.resources,
        )


class CouplingSolution(StrictModule):
    """Accepted coupled trajectory and per-window physical evidence."""

    final_state: CouplingState
    successful: Array
    retained_times: Array
    retained_participant_states: tuple[Any, ...]
    retained_exchange_values: tuple[Any, ...]
    retained_valid: Array
    statuses: Array
    converged: Array
    exchange_residual_norms: Array
    participant_statuses: Array
    participant_evaluations: Array
    coupling_iterations: Array
    problem_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    coupling_plan_id: str = eqx.field(static=True)
    rollout_plan_id: str = eqx.field(static=True)


class CouplingRolloutPlan(StrictModule, NonTrainableState):
    """State retention and deterministic replay across fixed coupling windows."""

    retention: CouplingRetentionPolicy = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    replay: FixedStepReplayPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        retention: CouplingRetentionPolicy = "final",
        checkpoint_stride: int = 1,
        replay: FixedStepReplayPolicy | None = None,
    ):
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown coupling retention policy.")
        stride = int(checkpoint_stride)
        if stride <= 0:
            raise ValueError("checkpoint_stride must be positive.")
        if retention != "checkpoints" and stride != 1:
            raise ValueError(
                "checkpoint_stride differs from one only for checkpoint retention."
            )
        replay_ = FixedStepReplayPolicy() if replay is None else replay
        if not isinstance(replay_, FixedStepReplayPolicy):
            raise TypeError("replay must be FixedStepReplayPolicy or None.")
        self.retention = retention
        self.checkpoint_stride = stride
        self.replay = replay_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupling-rollout-plan",
                "retention": retention,
                "checkpoint_stride": stride,
                "replay": replay_.policy_id,
            }
        )

    def rollout(
        self,
        prepared: PreparedCoupling,
        /,
        *,
        window_count: int,
        window_size: float,
        args: Any = None,
    ) -> CouplingSolution:
        if not isinstance(prepared, PreparedCoupling):
            raise TypeError("prepared must be PreparedCoupling.")
        count = int(window_count)
        if count <= 0:
            raise ValueError("window_count must be positive.")
        size = jnp.asarray(window_size, dtype=prepared.reference_state.time.dtype)
        exchange_count = len(prepared.exchanges)
        participant_count = len(prepared.subsystems)
        residual_dtype = prepared.reference_state.time.dtype

        def step(carry, window_index):
            state, active, terminal_status = carry

            def execute(_):
                result = advance_coupling_window(prepared, state, size, args)
                return (
                    result.accepted_state,
                    result.successful,
                    result.status,
                    result.converged,
                    result.diagnostics.exchange_residual_norms,
                    result.diagnostics.participant_statuses,
                    result.diagnostics.participant_evaluations,
                    result.diagnostics.coupling_iterations,
                )

            def skip(_):
                return (
                    state,
                    jnp.asarray(False),
                    terminal_status,
                    jnp.asarray(False),
                    jnp.full((exchange_count,), jnp.nan, dtype=residual_dtype),
                    jnp.full((participant_count,), -1, dtype=jnp.int32),
                    jnp.zeros((participant_count,), dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                )

            (
                accepted,
                successful,
                status,
                converged,
                residuals,
                participant_statuses,
                participant_evaluations,
                iterations,
            ) = jax.lax.cond(active, execute, skip, operand=None)
            next_active = active & successful
            next_status = jnp.where(active, status, terminal_status)
            next_carry = (accepted, next_active, next_status)
            payload = (
                successful,
                status,
                converged,
                residuals,
                participant_statuses,
                participant_evaluations,
                iterations,
            )
            return next_carry, payload

        indices = jnp.arange(count, dtype=jnp.int32)
        initial_carry = (
            prepared.reference_state,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )

        if self.retention == "trajectory":

            def trajectory_step(carry, window_index):
                next_carry, payload = step(carry, window_index)
                return next_carry, (next_carry[0], *payload)

            (final_state, final_success, _), payload = checkpointed_scan(
                trajectory_step,
                initial_carry,
                indices,
                length=count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
            )
            (
                states,
                valid,
                statuses,
                converged,
                residuals,
                participant_statuses,
                participant_evaluations,
                iterations,
            ) = payload
            retained_participant_states = tuple(
                _prepend(initial, values)
                for initial, values in zip(
                    prepared.reference_state.participant_states,
                    states.participant_states,
                    strict=True,
                )
            )
            retained_exchange_values = tuple(
                _prepend(initial, values)
                for initial, values in zip(
                    prepared.reference_state.exchange_values,
                    states.exchange_values,
                    strict=True,
                )
            )
            retained_valid = jnp.concatenate((jnp.asarray([True]), valid), axis=0)
            retained_times = jnp.asarray(
                prepared.reference_state.time, dtype=size.dtype
            ) + size * jnp.arange(count + 1)
        elif self.retention == "final":
            (final_state, final_success, _), payload = checkpointed_scan(
                step,
                initial_carry,
                indices,
                length=count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
            )
            (
                valid,
                statuses,
                converged,
                residuals,
                participant_statuses,
                participant_evaluations,
                iterations,
            ) = payload
            retained_participant_states = tuple(
                _singleton(value) for value in final_state.participant_states
            )
            retained_exchange_values = tuple(
                _singleton(value) for value in final_state.exchange_values
            )
            retained_valid = final_success[None]
            retained_times = final_state.time[None]
        else:
            saved_indices = tuple(range(0, count + 1, self.checkpoint_stride))
            if saved_indices[-1] != count:
                saved_indices = (*saved_indices, count)
            save_after_window = np.zeros((count,), dtype=bool)
            for endpoint in saved_indices[1:]:
                save_after_window[endpoint - 1] = True
            save_mask = jnp.asarray(save_after_window)
            participant_buffers = tuple(
                jax.tree.map(
                    lambda leaf: (
                        jnp.zeros((len(saved_indices), *leaf.shape), dtype=leaf.dtype)
                        .at[0]
                        .set(leaf)
                    ),
                    value,
                )
                for value in prepared.reference_state.participant_states
            )
            exchange_buffers = tuple(
                jax.tree.map(
                    lambda leaf: (
                        jnp.zeros((len(saved_indices), *leaf.shape), dtype=leaf.dtype)
                        .at[0]
                        .set(leaf)
                    ),
                    value,
                )
                for value in prepared.reference_state.exchange_values
            )
            retained_valid = jnp.zeros((len(saved_indices),), dtype=bool).at[0].set(True)

            def checkpoint_step(carry, window_index):
                base_carry, participants, exchanges, valid_buffer, cursor = carry
                next_carry, payload = step(base_carry, window_index)
                accepted, successful, _ = next_carry

                def store(values):
                    participant_values, exchange_values, validity, current = values
                    participant_values = tuple(
                        jax.tree.map(
                            lambda buffer, value: buffer.at[current].set(value),
                            buffer,
                            value,
                        )
                        for buffer, value in zip(
                            participant_values,
                            accepted.participant_states,
                            strict=True,
                        )
                    )
                    exchange_values = tuple(
                        jax.tree.map(
                            lambda buffer, value: buffer.at[current].set(value),
                            buffer,
                            value,
                        )
                        for buffer, value in zip(
                            exchange_values,
                            accepted.exchange_values,
                            strict=True,
                        )
                    )
                    validity = validity.at[current].set(successful)
                    return participant_values, exchange_values, validity, current + 1

                participants, exchanges, valid_buffer, cursor = jax.lax.cond(
                    save_mask[window_index],
                    store,
                    lambda values: values,
                    (participants, exchanges, valid_buffer, cursor),
                )
                return (
                    next_carry,
                    participants,
                    exchanges,
                    valid_buffer,
                    cursor,
                ), payload

            checkpoint_carry = (
                initial_carry,
                participant_buffers,
                exchange_buffers,
                retained_valid,
                jnp.asarray(1, dtype=jnp.int32),
            )
            result_carry, payload = checkpointed_scan(
                checkpoint_step,
                checkpoint_carry,
                indices,
                length=count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
            )
            (
                (final_state, final_success, _),
                retained_participant_states,
                retained_exchange_values,
                retained_valid,
                _,
            ) = result_carry
            (
                valid,
                statuses,
                converged,
                residuals,
                participant_statuses,
                participant_evaluations,
                iterations,
            ) = payload
            retained_times = jnp.asarray(
                prepared.reference_state.time, dtype=size.dtype
            ) + size * jnp.asarray(saved_indices, dtype=size.dtype)

        return CouplingSolution(
            final_state=final_state,
            successful=final_success,
            retained_times=retained_times,
            retained_participant_states=retained_participant_states,
            retained_exchange_values=retained_exchange_values,
            retained_valid=retained_valid,
            statuses=statuses,
            converged=converged,
            exchange_residual_norms=residuals,
            participant_statuses=participant_statuses,
            participant_evaluations=participant_evaluations,
            coupling_iterations=iterations,
            problem_id=prepared.problem_id,
            graph_id=prepared.graph_id,
            coupling_plan_id=prepared.plan_id,
            rollout_plan_id=self.plan_id,
        )


def solve_coupling(
    problem: CouplingProblem,
    /,
    *,
    rollout: CouplingRolloutPlan | None = None,
) -> CouplingSolution:
    """Prepare and execute one fixed-window native coupling problem."""

    if not isinstance(problem, CouplingProblem):
        raise TypeError("problem must be CouplingProblem.")
    rollout_ = CouplingRolloutPlan() if rollout is None else rollout
    if not isinstance(rollout_, CouplingRolloutPlan):
        raise TypeError("rollout must be CouplingRolloutPlan or None.")
    return rollout_.rollout(
        problem.prepare(),
        window_count=problem.window_count,
        window_size=problem.window_size,
        args=problem.args,
    )


__all__ = [
    "CouplingProblem",
    "CouplingRetentionPolicy",
    "CouplingRolloutPlan",
    "CouplingSolution",
    "solve_coupling",
]
