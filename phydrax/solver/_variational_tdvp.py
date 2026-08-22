#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..linalg import (
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    NullspacePolicy,
    solve,
)
from ..operators.quantum import ComplexParameterMode
from ._variational_monte_carlo import (
    _score_geometry,
    _validate_model_coordinates,
    _validate_state_compatibility,
    evaluate_variational_monte_carlo,
    VariationalMonteCarloEstimate,
    VariationalMonteCarloProblem,
    VariationalMonteCarloState,
    VMC_LINEAR_FAILURE,
    vmc_status_name,
    VMC_SUCCESS,
)


TDVPMode: TypeAlias = Literal["real-time", "imaginary-time"]


class VariationalTDVPPolicy(StrictModule):
    """Fixed-step stochastic TDVP policy over persistent Markov chains."""

    mode: TDVPMode = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    draws_per_step: int = eqx.field(static=True)
    transitions_per_draw: int = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    final_evaluation_draws: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    max_velocity_norm: float | None = eqx.field(static=True)
    energy_imag_tolerance: float = eqx.field(static=True)
    failure_mode: Literal["raise", "record"] = eqx.field(static=True)
    final_chain_diagnostics: bool = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    nullspace_policy: NullspacePolicy | None

    def __init__(
        self,
        mode: TDVPMode,
        /,
        *,
        num_steps: int,
        step_size: float,
        draws_per_step: int,
        transitions_per_draw: int = 1,
        warmup_steps: int = 0,
        final_evaluation_draws: int | None = None,
        damping: float = 1e-3,
        max_velocity_norm: float | None = None,
        energy_imag_tolerance: float = 1e-8,
        failure_mode: Literal["raise", "record"] = "raise",
        final_chain_diagnostics: bool = True,
        linear_policy: LinearSolvePolicy | None = None,
        nullspace_policy: NullspacePolicy | None = None,
    ):
        if mode not in ("real-time", "imaginary-time"):
            raise ValueError("mode must be 'real-time' or 'imaginary-time'.")
        steps = int(num_steps)
        draws = int(draws_per_step)
        transitions = int(transitions_per_draw)
        warmup = int(warmup_steps)
        final_draws = (
            draws if final_evaluation_draws is None else int(final_evaluation_draws)
        )
        size = float(step_size)
        damping_ = float(damping)
        tolerance = float(energy_imag_tolerance)
        if steps < 0:
            raise ValueError("num_steps must be non-negative.")
        if draws <= 0 or final_draws <= 0:
            raise ValueError("TDVP draw counts must be positive.")
        if transitions <= 0 or warmup < 0:
            raise ValueError(
                "transitions_per_draw must be positive and warmup_steps non-negative."
            )
        if not isfinite(size) or size <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_imag_tolerance must be finite and non-negative.")
        if max_velocity_norm is None:
            velocity_limit = None
        else:
            velocity_limit = float(max_velocity_norm)
            if not isfinite(velocity_limit) or velocity_limit <= 0.0:
                raise ValueError("max_velocity_norm must be finite and positive.")
        if failure_mode not in ("raise", "record"):
            raise ValueError("failure_mode must be 'raise' or 'record'.")
        if linear_policy is not None and not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if nullspace_policy is not None and not isinstance(
            nullspace_policy, NullspacePolicy
        ):
            raise TypeError("nullspace_policy must be a NullspacePolicy or None.")
        self.mode = mode
        self.num_steps = steps
        self.step_size = size
        self.draws_per_step = draws
        self.transitions_per_draw = transitions
        self.warmup_steps = warmup
        self.final_evaluation_draws = final_draws
        self.damping = damping_
        self.max_velocity_norm = velocity_limit
        self.energy_imag_tolerance = tolerance
        self.failure_mode = failure_mode
        self.final_chain_diagnostics = bool(final_chain_diagnostics)
        self.linear_policy = linear_policy
        self.nullspace_policy = nullspace_policy


class VariationalTDVPResult(StrictModule):
    """Parameter trajectory and stochastic evidence for one TDVP evolution."""

    final_state: VariationalMonteCarloState
    final_estimate: VariationalMonteCarloEstimate
    times: Array
    parameter_trajectory: Array
    energy_history: Array
    variance_history: Array
    acceptance_history: Array
    velocity_norm_history: Array
    status_history: Array
    linear_results: tuple[LinearSolveResult, ...]
    root_key: Array
    mode: TDVPMode = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    completed_steps: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.final_estimate.successful & jnp.all(
            self.status_history == VMC_SUCCESS
        )


def _tdvp_force(
    score: JacobianLinearOperator,
    local_energy: Array,
    mean_energy: Array,
    parameter_mode: ComplexParameterMode,
    evolution_mode: TDVPMode,
    /,
) -> Array:
    residual = jnp.asarray(local_energy).reshape((-1,)) - mean_energy
    count = int(residual.shape[0])
    if parameter_mode == "holomorphic":
        force = jnp.asarray(score.adjoint_mv(residual / count))
        return -1j * force if evolution_mode == "real-time" else -force
    if evolution_mode == "real-time":
        cotangent = jnp.stack((jnp.imag(residual), -jnp.real(residual)), axis=-1) / count
    else:
        cotangent = -jnp.stack((jnp.real(residual), jnp.imag(residual)), axis=-1) / count
    return jnp.asarray(score.adjoint_mv(cotangent))


def _raise_tdvp(status: Array, role: str, /) -> None:
    raise RuntimeError(f"{role} failed with VMC status {vmc_status_name(status)}.")


def solve_variational_tdvp(
    problem: VariationalMonteCarloProblem,
    policy: VariationalTDVPPolicy,
    /,
    *,
    key: Key[Array, ""] | None = None,
    state: VariationalMonteCarloState | None = None,
) -> VariationalTDVPResult:
    """Evolve amplitude parameters by fixed-step real- or imaginary-time TDVP."""
    if not isinstance(problem, VariationalMonteCarloProblem):
        raise TypeError("problem must be a VariationalMonteCarloProblem.")
    if not isinstance(policy, VariationalTDVPPolicy):
        raise TypeError("policy must be a VariationalTDVPPolicy.")
    if state is None:
        if key is None:
            raise ValueError("A root key is required for a new TDVP run.")
        resolved_key = key
        current = problem.initial_state(key=resolved_key)
    else:
        if not isinstance(state, VariationalMonteCarloState):
            raise TypeError("state must be a VariationalMonteCarloState or None.")
        resolved_key = state.root_key if key is None else key
        if not jnp.array_equal(jr.key_data(resolved_key), jr.key_data(state.root_key)):
            raise ValueError("Resume key does not match the TDVP state root key.")
        current = state
    _validate_state_compatibility(problem, current)
    _validate_model_coordinates(problem, current)

    coordinate_history = [current.parameter_coordinates]
    energies: list[Array] = []
    variances: list[Array] = []
    acceptances: list[Array] = []
    velocity_norms: list[Array] = []
    statuses: list[Array] = []
    linear_results: list[LinearSolveResult] = []

    for _ in range(policy.num_steps):
        iteration = int(current.iteration)
        step_key = jr.fold_in(resolved_key, iteration)
        estimate, samples = evaluate_variational_monte_carlo(
            problem,
            current.model,
            current.markov_state,
            key=step_key,
            num_draws=policy.draws_per_step,
            steps_per_draw=policy.transitions_per_draw,
            warmup_steps=policy.warmup_steps if iteration == 0 else 0,
            energy_imag_tolerance=policy.energy_imag_tolerance,
        )
        energies.append(estimate.energy)
        variances.append(estimate.variance)
        acceptances.append(estimate.acceptance_rate)
        if not bool(estimate.successful):
            statuses.append(estimate.status)
            velocity_norms.append(jnp.asarray(jnp.nan))
            current = VariationalMonteCarloState(
                model=current.model,
                parameter_coordinates=current.parameter_coordinates,
                markov_state=samples.final_state,
                iteration=iteration,
                root_key=resolved_key,
            )
            if policy.failure_mode == "raise":
                _raise_tdvp(estimate.status, "TDVP estimation")
            break

        score, metric = _score_geometry(
            problem,
            current.parameter_coordinates,
            samples.samples,
            damping=policy.damping,
        )
        force = _tdvp_force(
            score,
            estimate.local.value,
            estimate.energy,
            problem.complex_parameter_mode,
            policy.mode,
        )
        linear = solve(
            LinearSystem(metric, nullspace_policy=policy.nullspace_policy),
            force,
            policy=policy.linear_policy,
        )
        linear_results.append(linear)
        velocity = jnp.asarray(linear.value)
        if not bool(jnp.all(linear.successful)) or not bool(
            jnp.all(jnp.isfinite(velocity))
        ):
            status = jnp.asarray(VMC_LINEAR_FAILURE, dtype=jnp.int32)
            statuses.append(status)
            velocity_norms.append(jnp.asarray(jnp.nan))
            current = VariationalMonteCarloState(
                model=current.model,
                parameter_coordinates=current.parameter_coordinates,
                markov_state=samples.final_state,
                iteration=iteration,
                root_key=resolved_key,
            )
            if policy.failure_mode == "raise":
                _raise_tdvp(status, "TDVP metric solve")
            break

        norm = jnp.linalg.norm(velocity)
        if policy.max_velocity_norm is not None:
            scale = jnp.minimum(1.0, policy.max_velocity_norm / jnp.maximum(norm, 1e-30))
            velocity = scale * velocity
            norm = jnp.linalg.norm(velocity)
        coordinates = current.parameter_coordinates + policy.step_size * velocity
        model = problem.model_from_coordinates(coordinates)
        statuses.append(jnp.asarray(VMC_SUCCESS, dtype=jnp.int32))
        velocity_norms.append(norm)
        coordinate_history.append(coordinates)
        current = VariationalMonteCarloState(
            model=model,
            parameter_coordinates=coordinates,
            markov_state=samples.final_state,
            iteration=iteration + 1,
            root_key=resolved_key,
        )

    final_estimate, _ = evaluate_variational_monte_carlo(
        problem,
        current.model,
        current.markov_state,
        key=jr.fold_in(resolved_key, 0x7D1F),
        num_draws=policy.final_evaluation_draws,
        steps_per_draw=policy.transitions_per_draw,
        energy_imag_tolerance=policy.energy_imag_tolerance,
        compute_chain_diagnostics=policy.final_chain_diagnostics,
    )
    completed = len(statuses)
    return VariationalTDVPResult(
        final_state=current,
        final_estimate=final_estimate,
        times=policy.step_size * jnp.arange(len(coordinate_history), dtype=float),
        parameter_trajectory=jnp.stack(coordinate_history),
        energy_history=jnp.stack(energies)
        if energies
        else jnp.empty((0,), dtype=complex),
        variance_history=jnp.stack(variances) if variances else jnp.empty((0,)),
        acceptance_history=jnp.stack(acceptances) if acceptances else jnp.empty((0,)),
        velocity_norm_history=jnp.stack(velocity_norms)
        if velocity_norms
        else jnp.empty((0,)),
        status_history=jnp.stack(statuses)
        if statuses
        else jnp.empty((0,), dtype=jnp.int32),
        linear_results=tuple(linear_results),
        root_key=resolved_key,
        mode=policy.mode,
        problem_id=problem.problem_id,
        completed_steps=completed,
    )


__all__ = [
    "TDVPMode",
    "VariationalTDVPPolicy",
    "VariationalTDVPResult",
    "solve_variational_tdvp",
]
