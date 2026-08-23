#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._temporal_precision import TemporalPrecisionPolicy
from ..operators.quantum import (
    ApproximationAxis,
    ApproximationQuantity,
    OpenSystemApproximationEvidence,
)


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
    geometry_precision: GeometryPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: StateVectorOperator,
        collapse_operators: Sequence[StateVectorOperator],
        initial_state: ArrayLike,
        /,
        *,
        geometry_precision: GeometryPrecisionPolicy | None = None,
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
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        state = jnp.asarray(initial_state)
        geometry_.validate_coordinates(state)
        if state.shape != (hamiltonian.dimension,):
            raise ValueError("Initial state dimension does not match the Hamiltonian.")
        norm = geometry_.norm(state)
        if not bool(jax.device_get(jnp.isfinite(norm) & (norm > 0.0))):
            raise ValueError("Initial state must have finite nonzero norm.")
        self.hamiltonian = hamiltonian
        self.collapse_operators = collapse
        self.initial_state = jnp.asarray(state / norm, dtype=state.dtype)
        self.geometry_precision = geometry_
        self.precision_evidence = geometry_.evidence_for(state)
        self.problem_id = str(problem_id)


class QuantumTrajectoryEnsemble(StrictModule):
    states: Array
    jump_channels: Array
    jump_mask: Array
    times: Array
    approximation: OpenSystemApproximationEvidence
    valid: Array
    temporal_precision: TemporalPrecisionPolicy
    geometry_precision: GeometryPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
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
        maximum_step_size: float = 0.1,
        maximum_statistical_error: float = 0.25,
        temporal_precision: TemporalPrecisionPolicy,
        geometry_precision: GeometryPrecisionPolicy,
    ):
        if not isinstance(temporal_precision, TemporalPrecisionPolicy):
            raise TypeError("temporal_precision must be TemporalPrecisionPolicy.")
        if not isinstance(geometry_precision, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy.")
        states_ = jnp.asarray(states)
        times_ = jnp.asarray(times)
        temporal_precision.validate_state(states_[0, 0])
        geometry_precision.validate_coordinates(states_[0, 0])
        norm_residual = geometry_precision.decision(
            jnp.max(jnp.abs(geometry_precision.norm(states_, axis=-1) - 1.0))
        )
        self.states = temporal_precision.output(states_)
        self.jump_channels = jnp.asarray(jump_channels, dtype=jnp.int32)
        self.jump_mask = jnp.asarray(jump_mask, dtype=bool)
        self.times = times_
        self.valid = jnp.all(jnp.isfinite(self.states)) & (norm_residual <= 1e-6)
        statistical_error = 1.0 / jnp.sqrt(float(self.states.shape[0]))
        self.temporal_precision = temporal_precision
        self.geometry_precision = geometry_precision
        self.precision_evidence = temporal_precision.evidence_for(
            states_[0, 0],
            times_[0],
            children={
                "ensemble-reduction": geometry_precision.evidence_for(states_[0, 0])
            },
        )
        self.approximation = OpenSystemApproximationEvidence(
            "quantum-trajectory-ensemble",
            (
                ApproximationAxis("trajectory-count", self.states.shape[0]),
                ApproximationAxis("time-step", step_size, units="time"),
            ),
            (
                ApproximationQuantity(
                    "time-step",
                    temporal_precision.decision(jnp.asarray(step_size)),
                    maximum_step_size,
                    units="time",
                    norm_id="absolute",
                    estimate_kind="estimate",
                ),
                ApproximationQuantity(
                    "monte-carlo-standard-error-scale",
                    geometry_precision.decision(statistical_error),
                    maximum_statistical_error,
                    units="dimensionless",
                    norm_id="inverse-sqrt-sample-count",
                    estimate_kind="statistical",
                    confidence=0.682689,
                ),
            ),
            execution_valid=self.valid,
            precision_evidence=self.precision_evidence,
            precision_policy_ids=(
                temporal_precision.policy_id,
                geometry_precision.policy_id,
            ),
        )
        self.problem_id = str(problem_id)

    def observable(self, operator: StateVectorOperator, /) -> tuple[Array, Array]:
        values = jax.vmap(
            jax.vmap(
                lambda state: jnp.real(
                    self.geometry_precision.sum(jnp.conj(state) * operator(state))
                )
            )
        )(self.states)
        mean = self.geometry_precision.sum(values, axis=0) / values.shape[0]
        centered = self.geometry_precision.accumulation(values - mean)
        variance = (
            self.geometry_precision.sum(
                jnp.real(jnp.conj(centered) * centered),
                axis=0,
            )
            / values.shape[0]
        )
        error = self.geometry_precision.decision(jnp.sqrt(variance / values.shape[0]))
        return self.geometry_precision.decision(mean), error

    def empirical_density(self) -> Array:
        final = self.states[:, -1, :]
        projectors = jax.vmap(lambda state: state[:, None] * jnp.conj(state[None, :]))(
            final
        )
        return self.geometry_precision.output(
            self.geometry_precision.sum(projectors, axis=0) / final.shape[0]
        )


def _trajectory(
    problem: QuantumJumpProblem,
    key: Array,
    step: Array,
    count: int,
    temporal_precision: TemporalPrecisionPolicy,
    geometry_precision: GeometryPrecisionPolicy,
):
    channel_count = len(problem.collapse_operators)
    decision_address = SampleAddress(
        "quantum-trajectory",
        "fixed-step-jump-decision",
        target=problem.problem_id,
        role="decision",
    )
    channel_address = SampleAddress(
        "quantum-trajectory",
        "fixed-step-jump-channel",
        target=problem.problem_id,
        role="channel",
    )

    def advance(state, index):
        decision_key = derive_key(key, decision_address, index)
        channel_key = derive_key(key, channel_address, index)
        collapsed = temporal_precision.stage(
            jnp.stack([operator(state) for operator in problem.collapse_operators])
        )
        collapsed_ = geometry_precision.accumulation(collapsed)
        rates = geometry_precision.decision(
            jnp.real(
                geometry_precision.sum(
                    jnp.conj(collapsed_) * collapsed_,
                    axis=1,
                )
            )
        )
        probabilities = temporal_precision.decision(step * rates)
        total = temporal_precision.decision(geometry_precision.sum(probabilities))
        probabilities = eqx.error_if(
            probabilities,
            total > 0.1,
            "Fixed-step jump probability exceeds the 0.1 validity limit.",
        )
        jump = jax.random.uniform(decision_key) < total
        safe_total = jnp.maximum(total, jnp.finfo(probabilities.dtype).tiny)
        channel = jax.random.categorical(
            channel_key,
            jnp.log(jnp.maximum(probabilities / safe_total, 1e-30)),
        )
        selected = collapsed[jnp.minimum(channel, max(channel_count - 1, 0))]
        jump_state = jnp.asarray(
            selected / jnp.maximum(geometry_precision.norm(selected), 1e-30),
            dtype=state.dtype,
        )
        effective = temporal_precision.stage(-1j * problem.hamiltonian(state))
        for operator, collapsed_state in zip(
            problem.collapse_operators,
            collapsed,
            strict=True,
        ):
            effective = effective - 0.5 * temporal_precision.stage(
                operator.adjoint(collapsed_state)
            )
        no_jump = jnp.asarray(
            state + step * temporal_precision.accumulation(effective),
            dtype=state.dtype,
        )
        no_jump = jnp.asarray(
            no_jump / geometry_precision.norm(no_jump),
            dtype=state.dtype,
        )
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
    temporal_precision: TemporalPrecisionPolicy | None = None,
    geometry_precision: GeometryPrecisionPolicy | None = None,
) -> QuantumTrajectoryEnsemble:
    if not isinstance(problem, QuantumJumpProblem):
        raise TypeError("problem must be QuantumJumpProblem.")
    temporal_ = (
        TemporalPrecisionPolicy() if temporal_precision is None else temporal_precision
    )
    geometry_ = (
        problem.geometry_precision if geometry_precision is None else geometry_precision
    )
    if not isinstance(temporal_, TemporalPrecisionPolicy):
        raise TypeError("temporal_precision must be TemporalPrecisionPolicy or None.")
    if not isinstance(geometry_, GeometryPrecisionPolicy):
        raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
    temporal_.validate_state(problem.initial_state)
    geometry_.validate_coordinates(problem.initial_state)
    step = temporal_.coefficient(
        jnp.asarray(step_size, dtype=problem.initial_state.real.dtype)
    ).reshape(())
    count = int(steps)
    trajectories = int(trajectory_count)
    if count < 0 or trajectories < 1 or float(step) <= 0.0:
        raise ValueError("Trajectory count, steps, and step size must be positive.")
    initial_collapsed = jnp.stack(
        [operator(problem.initial_state) for operator in problem.collapse_operators]
    )
    initial_rates = jnp.real(
        jnp.einsum("ki,ki->k", jnp.conj(initial_collapsed), initial_collapsed)
    )
    if float(step * jnp.sum(initial_rates)) > 0.1:
        raise ValueError("Fixed-step jump probability exceeds the 0.1 validity limit.")
    keys = jax.random.split(key, trajectories)
    states, channels, masks = jax.vmap(
        lambda local_key: _trajectory(
            problem,
            local_key,
            step,
            count,
            temporal_,
            geometry_,
        )
    )(keys)
    return QuantumTrajectoryEnsemble(
        states,
        channels,
        masks,
        step * jnp.arange(count + 1),
        step_size=step,
        problem_id=problem.problem_id,
        temporal_precision=temporal_,
        geometry_precision=geometry_,
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
