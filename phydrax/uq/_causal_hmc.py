#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, NamedTuple, TypeAlias

import blackjax.mcmc.hmc as blackjax_hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
from blackjax.mcmc.trajectory import hmc_energy
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from .._strict import StrictModule
from ..nonlinear import (
    CausalLevenbergMarquardt,
    CausalLinearizationPolicy,
    CausalNewton,
    CausalRecurrenceProblem,
    NonlinearStatus,
    NonlinearTermination,
    solve_causal_recurrence,
)


CausalHMCLinearization: TypeAlias = Literal["dense-exact", "pair-hutchinson"]
CausalHMCFailurePolicy: TypeAlias = Literal["raise", "sequential"]


class CausalHMCConfig(StrictModule):
    """Static causal velocity-Verlet solve and explicit failure controls."""

    trajectory_block_size: int = eqx.field(static=True)
    linearization: CausalHMCLinearization = eqx.field(static=True)
    probe_count: int = eqx.field(static=True)
    absolute_residual: float = eqx.field(static=True)
    relative_residual: float = eqx.field(static=True)
    maximum_outer_iterations: int = eqx.field(static=True)
    initial_damping: float = eqx.field(static=True)
    failure_policy: CausalHMCFailurePolicy = eqx.field(static=True)
    maximum_dense_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        trajectory_block_size: int = 128,
        linearization: CausalHMCLinearization = "pair-hutchinson",
        probe_count: int = 2,
        absolute_residual: float = 1e-6,
        relative_residual: float = 1e-6,
        maximum_outer_iterations: int = 128,
        initial_damping: float = 1e-3,
        maximum_dense_dimension: int = 512,
        failure_policy: CausalHMCFailurePolicy = "raise",
    ):
        block_size = int(trajectory_block_size)
        probes = int(probe_count)
        iterations = int(maximum_outer_iterations)
        dense_cap = int(maximum_dense_dimension)
        if block_size < 1:
            raise ValueError("trajectory_block_size must be positive.")
        if linearization not in ("dense-exact", "pair-hutchinson"):
            raise ValueError("Unknown causal HMC linearization.")
        if probes < 1:
            raise ValueError("probe_count must be positive.")
        absolute = float(absolute_residual)
        relative = float(relative_residual)
        damping = float(initial_damping)
        if any(not isfinite(value) or value < 0.0 for value in (absolute, relative)):
            raise ValueError(
                "Causal HMC residual tolerances must be finite and nonnegative."
            )
        if iterations < 1:
            raise ValueError("maximum_outer_iterations must be positive.")
        if dense_cap < 1:
            raise ValueError("maximum_dense_dimension must be positive.")
        if not isfinite(damping) or damping <= 0.0:
            raise ValueError("initial_damping must be positive and finite.")
        if failure_policy not in ("raise", "sequential"):
            raise ValueError("failure_policy must be 'raise' or 'sequential'.")
        self.trajectory_block_size = block_size
        self.linearization = linearization
        self.probe_count = probes
        self.absolute_residual = absolute
        self.relative_residual = relative
        self.maximum_outer_iterations = iterations
        self.initial_damping = damping
        self.maximum_dense_dimension = dense_cap
        self.failure_policy = failure_policy

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "trajectory_block_size": self.trajectory_block_size,
            "linearization": self.linearization,
            "probe_count": self.probe_count,
            "absolute_residual": self.absolute_residual,
            "relative_residual": self.relative_residual,
            "maximum_outer_iterations": self.maximum_outer_iterations,
            "initial_damping": self.initial_damping,
            "maximum_dense_dimension": self.maximum_dense_dimension,
            "failure_policy": self.failure_policy,
        }


class CausalNUTSConfig(StrictModule):
    """Fixed-capacity dynamic-tree causal NUTS controls."""

    recurrence: CausalHMCConfig
    max_num_doublings: int = eqx.field(static=True)
    max_trajectory_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_num_doublings: int = 10,
        max_trajectory_capacity: int | None = None,
        recurrence: CausalHMCConfig | None = None,
    ):
        doublings = int(max_num_doublings)
        if doublings <= 0:
            raise ValueError("max_num_doublings must be positive.")
        capacity = (
            2**doublings
            if max_trajectory_capacity is None
            else int(max_trajectory_capacity)
        )
        if capacity != 2**doublings:
            raise ValueError("max_trajectory_capacity must equal 2**max_num_doublings.")
        recurrence_ = (
            CausalHMCConfig(linearization="dense-exact")
            if recurrence is None
            else recurrence
        )
        if not isinstance(recurrence_, CausalHMCConfig):
            raise TypeError("recurrence must be CausalHMCConfig or None.")
        if recurrence_.linearization != "dense-exact":
            raise ValueError("Causal NUTS requires deterministic dense-exact gates.")
        self.recurrence = recurrence_
        self.max_num_doublings = doublings
        self.max_trajectory_capacity = capacity

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_num_doublings": self.max_num_doublings,
            "max_trajectory_capacity": self.max_trajectory_capacity,
            "recurrence": self.recurrence.as_dict(),
        }


class CausalHMCDiagnostics(StrictModule):
    """Production-draw causal trajectory convergence and fallback records."""

    converged: Array
    fallback_used: Array
    outer_iterations: Array
    maximum_residual: Array
    accepted_nonlinear_steps: Array
    rejected_nonlinear_steps: Array
    transition_evaluations: Array


class CausalHMCInfo(NamedTuple):
    momentum: Any
    acceptance_rate: Array
    is_accepted: Array
    is_divergent: Array
    energy: Array
    proposal: integrators.IntegratorState
    num_integration_steps: Array
    causal_converged: Array
    causal_fallback_used: Array
    causal_outer_iterations: Array
    causal_maximum_residual: Array
    causal_accepted_steps: Array
    causal_rejected_steps: Array
    causal_transition_evaluations: Array


def _sequential_block(
    integrator,
    phase,
    logdensity_fn,
    step_size,
    block_length: int,
    /,
):
    position, momentum = phase
    logdensity, gradient = jax.value_and_grad(logdensity_fn)(position)
    initial = integrators.IntegratorState(position, momentum, logdensity, gradient)
    final = jax.lax.fori_loop(
        0,
        block_length,
        lambda _, state: integrator(state, step_size),
        initial,
    )
    return final.position, final.momentum


def _pair_hutchinson_builder(
    logdensity_fn,
    inverse_mass_matrix: Array,
    step_size: Array,
    /,
):
    inverse_mass = jnp.asarray(inverse_mass_matrix)

    def build(_, previous, driver):
        position, momentum = previous
        flat_position, unravel = ravel_pytree(position)
        flat_momentum, _ = ravel_pytree(momentum)
        probes = driver["probes"]

        def flat_logdensity(value):
            return logdensity_fn(unravel(value))

        gradient_fn = jax.grad(flat_logdensity)
        gradient = gradient_fn(flat_position)

        def hessian_diagonal(location):
            def one_probe(probe):
                _, action = jax.jvp(
                    gradient_fn,
                    (location,),
                    (probe,),
                )
                return probe * action

            return jnp.mean(jax.vmap(one_probe)(probes), axis=0)

        first_hessian = hessian_diagonal(flat_position)
        half_momentum = flat_momentum + 0.5 * step_size * gradient
        next_position = flat_position + step_size * inverse_mass * half_momentum
        second_hessian = hessian_diagonal(next_position)
        momentum_position = 0.5 * step_size * first_hessian
        position_position = 1.0 + step_size * inverse_mass * momentum_position
        position_momentum = step_size * inverse_mass
        final_momentum_position = (
            momentum_position + 0.5 * step_size * second_hessian * position_position
        )
        final_momentum_momentum = (
            1.0 + 0.5 * step_size * second_hessian * position_momentum
        )
        dimension = flat_position.size
        matrix = jnp.zeros((2 * dimension, 2 * dimension), dtype=flat_position.dtype)
        matrix = matrix.at[:dimension, :dimension].set(jnp.diag(position_position))
        matrix = matrix.at[:dimension, dimension:].set(jnp.diag(position_momentum))
        matrix = matrix.at[dimension:, :dimension].set(jnp.diag(final_momentum_position))
        return matrix.at[dimension:, dimension:].set(jnp.diag(final_momentum_momentum))

    return build


def _causal_block(
    logdensity_fn,
    metric,
    inverse_mass_matrix,
    step_size,
    phase,
    block_length: int,
    probe_key,
    config: CausalHMCConfig,
    /,
):
    integrator = integrators.velocity_verlet(logdensity_fn, metric.kinetic_energy)
    dimension = int(jnp.asarray(inverse_mass_matrix).shape[0])
    if config.linearization == "pair-hutchinson":
        probes = jr.rademacher(
            probe_key,
            (block_length, config.probe_count, dimension),
            dtype=ravel_pytree(phase[0])[0].dtype,
        )
        drivers = {
            "step": jnp.arange(block_length, dtype=jnp.int32),
            "probes": probes,
        }
        builder = _pair_hutchinson_builder(
            logdensity_fn,
            inverse_mass_matrix,
            step_size,
        )
        method = CausalLevenbergMarquardt(
            linearization=CausalLinearizationPolicy(
                "fixed-block",
                block_builder=builder,
                linearization_id="velocity-verlet-pair-hutchinson",
            ),
            initial_damping=config.initial_damping,
        )
    else:
        drivers = jnp.arange(block_length, dtype=jnp.int32)
        method = CausalNewton()

    def transition(_, previous, driver):
        del driver
        position, momentum = previous
        logdensity, gradient = jax.value_and_grad(logdensity_fn)(position)
        state = integrators.IntegratorState(
            position,
            momentum,
            logdensity,
            gradient,
        )
        next_state = integrator(state, step_size)
        return next_state.position, next_state.momentum

    problem = CausalRecurrenceProblem(
        transition,
        phase,
        drivers,
        problem_id="causal-hmc-velocity-verlet-block",
    )
    result = solve_causal_recurrence(
        problem,
        method=method,
        termination=NonlinearTermination(
            absolute_residual=config.absolute_residual,
            relative_residual=config.relative_residual,
            maximum_steps=config.maximum_outer_iterations,
        ),
        probe_key=None,
    )
    causal_phase = result.final_state
    successful = result.status == int(NonlinearStatus.SUCCESS)
    if config.failure_policy == "raise":
        checked = eqx.error_if(
            result.flat_states,
            ~successful,
            "Causal HMC trajectory block failed to converge.",
        )
        causal_phase = problem.unravel_state(checked[-1])
        final_phase = causal_phase
        fallback = jnp.asarray(False)
    else:
        sequential_phase = _sequential_block(
            integrator,
            phase,
            logdensity_fn,
            step_size,
            block_length,
        )
        final_phase = jax.tree.map(
            lambda causal, sequential: jnp.where(
                successful,
                causal,
                sequential,
            ),
            causal_phase,
            sequential_phase,
        )
        fallback = ~successful
    diagnostics = result.diagnostics
    return final_phase, (
        successful,
        fallback,
        diagnostics.iteration_count,
        jnp.max(jnp.abs(result.flat_residuals)),
        diagnostics.accepted_steps,
        diagnostics.rejected_steps,
        diagnostics.transition_evaluations,
    )


def build_causal_hmc_kernel(
    config: CausalHMCConfig,
    /,
    *,
    divergence_threshold: float = 1000.0,
):
    """Build a BlackJAX-compatible fixed-trajectory causal HMC kernel."""

    if not isinstance(config, CausalHMCConfig):
        raise TypeError("config must be a CausalHMCConfig.")
    threshold = float(divergence_threshold)
    if not isfinite(threshold) or threshold <= 0.0:
        raise ValueError("divergence_threshold must be positive and finite.")

    def kernel(
        rng_key,
        state,
        logdensity_fn,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
    ):
        inverse_mass = jnp.asarray(inverse_mass_matrix)
        if inverse_mass.ndim not in (1, 2):
            raise ValueError(
                "Causal HMC inverse mass must be diagonal or a square dense matrix."
            )
        dimension = int(inverse_mass.shape[0])
        if inverse_mass.ndim == 2:
            if inverse_mass.shape != (dimension, dimension):
                raise ValueError("Dense inverse mass must be square.")
            if config.linearization == "pair-hutchinson":
                raise ValueError(
                    "pair-hutchinson remains diagonal-only; use dense-exact "
                    "linearization for a non-diagonal inverse mass."
                )
            if dimension > config.maximum_dense_dimension:
                raise MemoryError(
                    "Dense causal HMC inverse mass exceeds maximum_dense_dimension."
                )
            factor = jnp.linalg.cholesky(inverse_mass)
            inverse_mass = eqx.error_if(
                inverse_mass,
                jnp.any(~jnp.isfinite(factor))
                | jnp.any(jnp.real(jnp.diag(factor)) <= 0.0),
                "Dense causal HMC inverse mass must be positive definite.",
            )
        steps = int(num_integration_steps)
        if steps < 1:
            raise ValueError("num_integration_steps must be positive.")
        metric = metrics.default_metric(inverse_mass)
        key_momentum, key_integrator = jr.split(rng_key, 2)
        momentum: Any = metric.sample_momentum(key_momentum, state.position)
        phase: tuple[Any, Any] = (state.position, momentum)
        block_records = []
        block_start = 0
        block_index = 0
        while block_start < steps:
            block_length = min(config.trajectory_block_size, steps - block_start)
            block_probe_key = jr.fold_in(key_integrator, 0xCA551 + block_index)
            phase, record = _causal_block(
                logdensity_fn,
                metric,
                inverse_mass,
                step_size,
                phase,
                block_length,
                block_probe_key,
                config,
            )
            block_records.append(record)
            block_start += block_length
            block_index += 1

        position, final_momentum = phase
        logdensity, gradient = jax.value_and_grad(logdensity_fn)(position)
        end_state = blackjax_hmc.flip_momentum(
            integrators.IntegratorState(
                position,
                final_momentum,
                logdensity,
                gradient,
            )
        )
        initial_integrator_state = integrators.IntegratorState(
            state.position,
            momentum,
            state.logdensity,
            state.logdensity_grad,
        )
        energy_fn = hmc_energy(metric.kinetic_energy)
        initial_energy = energy_fn(initial_integrator_state)
        new_energy = energy_fn(end_state)
        delta_energy = safe_energy_diff(initial_energy, new_energy)
        is_divergent: Array = jnp.asarray(-delta_energy > threshold)
        sampled, acceptance_info = static_binomial_sampling(
            key_integrator,
            delta_energy,
            initial_integrator_state,
            end_state,
        )
        is_accepted, acceptance_rate, _ = acceptance_info
        converged, fallback, iterations, residual, accepted, rejected, evaluations = (
            jax.tree.map(lambda *values: jnp.stack(values), *block_records)
        )
        next_state = blackjax_hmc.HMCState(
            sampled.position,
            sampled.logdensity,
            sampled.logdensity_grad,
        )
        info = CausalHMCInfo(
            momentum=momentum,
            acceptance_rate=acceptance_rate,
            is_accepted=is_accepted,
            is_divergent=is_divergent,
            energy=new_energy,
            proposal=end_state,
            num_integration_steps=jnp.asarray(steps, dtype=jnp.int32),
            causal_converged=jnp.all(converged),
            causal_fallback_used=jnp.any(fallback),
            causal_outer_iterations=jnp.max(iterations),
            causal_maximum_residual=jnp.max(residual),
            causal_accepted_steps=jnp.sum(accepted),
            causal_rejected_steps=jnp.sum(rejected),
            causal_transition_evaluations=jnp.sum(evaluations),
        )
        return next_state, info

    return kernel


__all__ = [
    "build_causal_hmc_kernel",
    "CausalHMCConfig",
    "CausalNUTSConfig",
    "CausalHMCDiagnostics",
    "CausalHMCFailurePolicy",
    "CausalHMCInfo",
    "CausalHMCLinearization",
]
