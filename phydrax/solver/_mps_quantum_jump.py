#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..tensor_network import MatrixProductState, NearestNeighborHamiltonian, tebd_step


class LocalMPSJump(StrictModule):
    operator: Array
    site: int = eqx.field(static=True)
    jump_id: str = eqx.field(static=True)

    def __init__(self, site: int, operator: ArrayLike, /, *, jump_id: str):
        value = jnp.asarray(operator)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Local jump operator must be square.")
        self.operator = value
        self.site = int(site)
        self.jump_id = str(jump_id)

    def apply(
        self, state: MatrixProductState, /, *, normalize: bool
    ) -> MatrixProductState:
        if not 0 <= self.site < state.site_count:
            raise ValueError("Jump site is outside the MPS.")
        tensor = state.tensors[self.site]
        if self.operator.shape[1] != tensor.shape[1]:
            raise ValueError("Jump physical dimension does not match the MPS site.")
        updated = jnp.einsum("oi,lir->lor", self.operator, tensor)
        tensors = list(state.tensors)
        tensors[self.site] = updated
        result = MatrixProductState(tuple(tensors))
        return result.normalized() if normalize else result

    def rate(self, state: MatrixProductState, /) -> Array:
        transformed = self.apply(state, normalize=False)
        return transformed.norm() ** 2


class MPSQuantumJumpProblem(StrictModule):
    hamiltonian: NearestNeighborHamiltonian
    jumps: tuple[LocalMPSJump, ...]
    initial_state: MatrixProductState
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: NearestNeighborHamiltonian,
        jumps: Sequence[LocalMPSJump],
        initial_state: MatrixProductState,
        /,
        *,
        problem_id: str = "mps-quantum-jump",
    ):
        jumps_ = tuple(jumps)
        if not jumps_:
            raise ValueError("At least one MPS jump operator is required.")
        if tuple(initial_state.physical_dimensions) != hamiltonian.physical_dimensions:
            raise ValueError("MPS and Hamiltonian dimensions differ.")
        self.hamiltonian = hamiltonian
        self.jumps = jumps_
        self.initial_state = initial_state.normalized()
        self.problem_id = str(problem_id)


class MPSQuantumTrajectoryResult(StrictModule):
    final_state: MatrixProductState
    jump_times: Array
    jump_channels: Array
    active_events: Array
    discarded_weight_history: Array
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        final_state: MatrixProductState,
        jump_times: ArrayLike,
        jump_channels: ArrayLike,
        active_events: ArrayLike,
        discarded_weight_history: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.final_state = final_state
        self.jump_times = jnp.asarray(jump_times)
        self.jump_channels = jnp.asarray(jump_channels, dtype=jnp.int32)
        self.active_events = jnp.asarray(active_events, dtype=bool)
        self.discarded_weight_history = jnp.asarray(discarded_weight_history)
        self.valid = (
            jnp.isfinite(final_state.norm())
            & (jnp.abs(final_state.norm() - 1.0) <= 1e-6)
            & jnp.all(jnp.isfinite(self.discarded_weight_history))
        )
        self.problem_id = str(problem_id)


def solve_mps_quantum_jump(
    problem: MPSQuantumJumpProblem,
    key: Array,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_bond_dimension: int,
    maximum_events: int = 128,
) -> MPSQuantumTrajectoryResult:
    step = jnp.asarray(step_size, dtype=float).reshape(())
    state = problem.initial_state
    times = jnp.zeros((maximum_events,), dtype=step.dtype)
    channels = -jnp.ones((maximum_events,), dtype=jnp.int32)
    active = jnp.zeros((maximum_events,), dtype=bool)
    discarded = []
    event_count = 0
    threshold = jax.random.uniform(key)
    accumulated_hazard = jnp.asarray(0.0)
    for index in range(int(steps)):
        state, evidence = tebd_step(
            state,
            problem.hamiltonian,
            step,
            maximum_bond_dimension=maximum_bond_dimension,
            order=2,
        )
        discarded.append(evidence.cumulative_discarded_weight)
        rates = jnp.stack([jump.rate(state) for jump in problem.jumps])
        total = jnp.sum(rates)
        accumulated_hazard = accumulated_hazard + step * total
        crossing = 1.0 - jnp.exp(-accumulated_hazard) >= threshold
        if bool(jax.device_get(crossing)) and event_count < maximum_events:
            local_key = jax.random.fold_in(key, event_count + 1)
            channel = jax.random.categorical(
                local_key, jnp.log(jnp.maximum(rates / total, 1e-30))
            )
            state = problem.jumps[int(channel)].apply(state, normalize=True)
            times = times.at[event_count].set((index + 1) * step)
            channels = channels.at[event_count].set(channel)
            active = active.at[event_count].set(True)
            event_count += 1
            accumulated_hazard = jnp.asarray(0.0)
            threshold = jax.random.uniform(jax.random.fold_in(key, event_count + 1000))
    return MPSQuantumTrajectoryResult(
        state,
        times,
        channels,
        active,
        jnp.stack(discarded) if discarded else jnp.zeros((0,)),
        problem_id=problem.problem_id,
    )


__all__ = [
    "LocalMPSJump",
    "MPSQuantumJumpProblem",
    "MPSQuantumTrajectoryResult",
    "solve_mps_quantum_jump",
]
