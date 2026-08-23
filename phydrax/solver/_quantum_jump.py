#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import ApproximationAxis, OpenSystemApproximationEvidence


class StateVectorOperator(StrictModule):
    action_function: Callable[[Array], Array]
    adjoint_function: Callable[[Array], Array]
    dimension: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[Array], Array],
        adjoint_action: Callable[[Array], Array],
        dimension: int,
        /,
        *,
        operator_id: str,
    ):
        if not callable(action) or not callable(adjoint_action):
            raise TypeError("Operator and adjoint actions must be callable.")
        self.action_function = action
        self.adjoint_function = adjoint_action
        self.dimension = int(dimension)
        self.operator_id = str(operator_id)

    @classmethod
    def from_matrix(cls, matrix: ArrayLike, /, *, operator_id: str):
        value = jnp.asarray(matrix)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Operator matrix must be square.")
        return cls(
            lambda state: value @ state,
            lambda state: jnp.conj(value.T) @ state,
            value.shape[0],
            operator_id=operator_id,
        )

    def __call__(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.dimension,):
            raise ValueError("State-vector shape does not match the operator.")
        result = jnp.asarray(self.action_function(value))
        if result.shape != value.shape:
            raise ValueError("Operator action must preserve state shape.")
        return result

    def adjoint(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.dimension,):
            raise ValueError("State-vector shape does not match the operator.")
        result = jnp.asarray(self.adjoint_function(value))
        if result.shape != value.shape:
            raise ValueError("Adjoint action must preserve state shape.")
        return result


class QuantumJumpProblem(StrictModule):
    hamiltonian: StateVectorOperator
    collapse_operators: tuple[StateVectorOperator, ...]
    initial_state: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: StateVectorOperator,
        collapse_operators: Sequence[StateVectorOperator],
        initial_state: ArrayLike,
        /,
        *,
        problem_id: str = "quantum-jump",
    ):
        if not isinstance(hamiltonian, StateVectorOperator):
            raise TypeError("hamiltonian must be a StateVectorOperator.")
        collapse = tuple(collapse_operators)
        if any(
            not isinstance(operator, StateVectorOperator)
            or operator.dimension != hamiltonian.dimension
            for operator in collapse
        ):
            raise ValueError("Collapse operators must share the Hamiltonian dimension.")
        state = jnp.asarray(initial_state)
        if state.shape != (hamiltonian.dimension,):
            raise ValueError("Initial state dimension does not match the Hamiltonian.")
        norm = jnp.linalg.norm(state)
        if not bool(jax.device_get(jnp.isfinite(norm) & (norm > 0.0))):
            raise ValueError("Initial state must have finite nonzero norm.")
        self.hamiltonian = hamiltonian
        self.collapse_operators = collapse
        self.initial_state = state / norm
        self.problem_id = str(problem_id)


class QuantumTrajectoryEnsemble(StrictModule):
    states: Array
    jump_channels: Array
    jump_mask: Array
    times: Array
    approximation: OpenSystemApproximationEvidence
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: ArrayLike,
        jump_channels: ArrayLike,
        jump_mask: ArrayLike,
        times: ArrayLike,
        /,
        *,
        step_size: ArrayLike,
        problem_id: str,
    ):
        self.states = jnp.asarray(states)
        self.jump_channels = jnp.asarray(jump_channels, dtype=jnp.int32)
        self.jump_mask = jnp.asarray(jump_mask, dtype=bool)
        self.times = jnp.asarray(times)
        norm_residual = jnp.max(jnp.abs(jnp.linalg.norm(self.states, axis=-1) - 1.0))
        self.valid = jnp.all(jnp.isfinite(self.states)) & (norm_residual <= 1e-6)
        self.approximation = OpenSystemApproximationEvidence(
            "quantum-trajectory-ensemble",
            (
                ApproximationAxis("trajectory-count", self.states.shape[0]),
                ApproximationAxis("time-step", step_size, units="time"),
            ),
            statistical_error=1.0 / jnp.sqrt(float(self.states.shape[0])),
            local_error=jnp.asarray(step_size),
            valid=self.valid,
        )
        self.problem_id = str(problem_id)

    def observable(self, operator: StateVectorOperator, /) -> tuple[Array, Array]:
        values = jax.vmap(
            jax.vmap(lambda state: jnp.real(jnp.vdot(state, operator(state))))
        )(self.states)
        return jnp.mean(values, axis=0), jnp.std(values, axis=0) / jnp.sqrt(
            float(values.shape[0])
        )

    def empirical_density(self) -> Array:
        final = self.states[:, -1, :]
        return jnp.mean(
            jax.vmap(lambda state: state[:, None] * jnp.conj(state[None, :]))(final),
            axis=0,
        )


def _trajectory(
    problem: QuantumJumpProblem,
    key: Array,
    step: Array,
    count: int,
):
    channel_count = len(problem.collapse_operators)

    def advance(state, index):
        local_key = jax.random.fold_in(key, index)
        collapsed = jnp.stack(
            [operator(state) for operator in problem.collapse_operators]
        )
        rates = jnp.real(jnp.einsum("ki,ki->k", jnp.conj(collapsed), collapsed))
        probabilities = step * rates
        total = jnp.sum(probabilities)
        jump = jax.random.uniform(local_key) < jnp.minimum(total, 1.0)
        safe_total = jnp.maximum(total, jnp.finfo(probabilities.dtype).tiny)
        channel = jax.random.categorical(
            local_key, jnp.log(jnp.maximum(probabilities / safe_total, 1e-30))
        )
        selected = collapsed[jnp.minimum(channel, max(channel_count - 1, 0))]
        jump_state = selected / jnp.maximum(jnp.linalg.norm(selected), 1e-30)
        effective = -1j * problem.hamiltonian(state)
        for operator, collapsed_state in zip(
            problem.collapse_operators, collapsed, strict=True
        ):
            effective = effective - 0.5 * operator.adjoint(collapsed_state)
        no_jump = state + step * effective
        no_jump = no_jump / jnp.linalg.norm(no_jump)
        next_state = jnp.where(jump, jump_state, no_jump)
        return next_state, (next_state, channel, jump)

    _, history = jax.lax.scan(advance, problem.initial_state, jnp.arange(count))
    states = jnp.concatenate((problem.initial_state[None, :], history[0]), axis=0)
    return states, history[1], history[2]


def solve_quantum_jump_ensemble(
    problem: QuantumJumpProblem,
    key: Array,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    trajectory_count: int,
) -> QuantumTrajectoryEnsemble:
    step = jnp.asarray(step_size, dtype=float).reshape(())
    count = int(steps)
    trajectories = int(trajectory_count)
    if count < 0 or trajectories < 1 or float(step) <= 0.0:
        raise ValueError("Trajectory count, steps, and step size must be positive.")
    keys = jax.random.split(key, trajectories)
    states, channels, masks = jax.vmap(
        lambda local_key: _trajectory(problem, local_key, step, count)
    )(keys)
    return QuantumTrajectoryEnsemble(
        states,
        channels,
        masks,
        step * jnp.arange(count + 1),
        step_size=step,
        problem_id=problem.problem_id,
    )


def amplitude_damping_trajectory_problem(
    damping_rate: float,
    initial_state: ArrayLike,
    /,
) -> QuantumJumpProblem:
    lowering = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return QuantumJumpProblem(
        StateVectorOperator.from_matrix(
            jnp.zeros((2, 2), dtype=complex), operator_id="zero-hamiltonian"
        ),
        (
            StateVectorOperator.from_matrix(
                jnp.sqrt(float(damping_rate)) * lowering,
                operator_id="amplitude-damping-jump",
            ),
        ),
        initial_state,
        problem_id="amplitude-damping-trajectories",
    )


__all__ = [
    "QuantumJumpProblem",
    "QuantumTrajectoryEnsemble",
    "StateVectorOperator",
    "amplitude_damping_trajectory_problem",
    "solve_quantum_jump_ensemble",
]
