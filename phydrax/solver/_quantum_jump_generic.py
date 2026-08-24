#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..stochastic import JumpProcess, PoissonClockRealization
from ._differential import DifferentialProblem
from ._jump import (
    JumpDifferentialProblem,
    JumpDifferentialSolution,
    solve_jump_differential,
)
from ._quantum_jump import QuantumJumpProblem


def quantum_jump_differential_problem(
    problem: QuantumJumpProblem,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
) -> JumpDifferentialProblem:
    """Adapt normalized quantum trajectories to the generic jump-differential solver."""
    if not isinstance(problem, QuantumJumpProblem):
        raise TypeError("problem must be a QuantumJumpProblem.")
    dimension = problem.initial_state.shape[0]

    def to_complex(state: Array) -> Array:
        return state[:dimension] + 1j * state[dimension:]

    def to_real(state: Array) -> Array:
        return jnp.concatenate((jnp.real(state), jnp.imag(state)))

    def rates(state: Array) -> Array:
        quantum_state = to_complex(state)
        collapsed = jnp.stack(
            [operator(quantum_state) for operator in problem.collapse_operators]
        )
        return jnp.real(oe.contract("ki,ki->k", jnp.conj(collapsed), collapsed))

    def drift(time, state, args):
        del time, args
        quantum_state = to_complex(state)
        channel_rates = rates(state)
        result = -1j * problem.hamiltonian(quantum_state)
        for operator in problem.collapse_operators:
            result = result - 0.5 * operator.adjoint(operator(quantum_state))
        normalized_result = result + 0.5 * jnp.sum(channel_rates) * quantum_state
        return to_real(normalized_result)

    def jump(state, channel, mark, args):
        del mark, args
        quantum_state = to_complex(state)
        collapsed = jnp.stack(
            [operator(quantum_state) for operator in problem.collapse_operators]
        )
        selected = collapsed[channel]
        return to_real(selected / jnp.linalg.norm(selected))

    process = JumpProcess(
        lambda time, state, args: rates(state),
        jump,
        state_shape=(2 * dimension,),
        num_channels=len(problem.collapse_operators),
        process_id=f"{problem.problem_id}:generic-jumps",
    )
    initial_state = to_real(problem.initial_state)
    differential = DifferentialProblem(
        drift,
        initial_state,
        t0=t0,
        t1=t1,
        args=None,
    )
    return JumpDifferentialProblem(
        differential,
        process,
        process_id=process.process_id,
    )


def solve_quantum_jump_generic(
    problem: QuantumJumpProblem,
    key: Array,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    save_times: ArrayLike,
    trajectory_count: int = 1,
    maximum_events_per_channel: int = 64,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> JumpDifferentialSolution:
    hybrid = quantum_jump_differential_problem(problem, t0=t0, t1=t1)
    realization = PoissonClockRealization(
        key,
        len(problem.collapse_operators),
        support=(float(t0), float(t1)),
        max_events_per_channel=maximum_events_per_channel,
        sample_shape=(int(trajectory_count),),
        process_id=hybrid.process_id,
    )
    return solve_jump_differential(
        hybrid,
        realization,
        save_times=save_times,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
    )


__all__ = [
    "quantum_jump_differential_problem",
    "solve_quantum_jump_generic",
]
