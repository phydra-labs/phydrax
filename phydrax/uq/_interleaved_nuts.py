#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.mcmc import hmc, integrators, metrics, proposal, termination, trajectory
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree


_Proposal = cast(Any, proposal.Proposal)
_Trajectory = cast(Any, trajectory.Trajectory)
_IterativeUTurnState = cast(Any, termination.IterativeUTurnState)
_reorder_trajectories = cast(Any, trajectory.reorder_trajectories)


class InterleavedNUTSStats(NamedTuple):
    """Execution statistics for one interleaved sampling chunk."""

    num_scheduler_steps: Array


class _NUTSContinuation(NamedTuple):
    current_state: Any
    integrator_key: Array
    initial_energy: Array
    expansion_index: Array
    global_proposal: proposal.Proposal
    global_trajectory: trajectory.Trajectory
    termination_state: termination.IterativeUTurnState
    local_step: Array
    local_proposal: proposal.Proposal
    local_trajectory: trajectory.Trajectory


class _NUTSEmission(NamedTuple):
    state: Any
    logdensity: Array
    acceptance_rate: Array
    is_divergent: Array
    energy: Array
    num_integration_steps: Array
    num_trajectory_expansions: Array


class _NUTSBuffers(NamedTuple):
    position: PyTree[Array]
    logdensity: Array
    acceptance_rate: Array
    is_divergent: Array
    energy: Array
    num_integration_steps: Array
    num_trajectory_expansions: Array


class _SchedulerCarry(NamedTuple):
    continuations: _NUTSContinuation
    completed: Array
    buffers: _NUTSBuffers
    num_scheduler_steps: Array


def build_interleaved_nuts_advancer(
    logdensity_fn: Callable[[PyTree[Any]], Array],
    /,
    *,
    max_num_doublings: int,
    divergence_threshold: float = 1000.0,
) -> Callable:
    """Build one reusable, chunk-shaped interleaved NUTS executable."""
    doublings = int(max_num_doublings)
    if doublings <= 0:
        raise ValueError("max_num_doublings must be positive.")
    threshold = float(divergence_threshold)

    def run(current_states, step_sizes, inverse_mass_matrices, draw_keys):
        chains, count = draw_keys.shape

        def prepare_chain(state, keys, inverse_mass_matrix):
            metric = metrics.default_metric(inverse_mass_matrix)
            split_keys = jax.vmap(lambda key: jr.split(key, 2))(keys)
            momentum_keys = split_keys[:, 0]
            integrator_keys = split_keys[:, 1]
            momenta = jax.vmap(metric.sample_momentum, in_axes=(0, None))(
                momentum_keys,
                state.position,
            )
            kinetic_energies = jax.vmap(metric.kinetic_energy)(momenta)
            return momenta, kinetic_energies, integrator_keys

        momenta, kinetic_energies, integrator_keys = jax.vmap(prepare_chain)(
            current_states,
            draw_keys,
            inverse_mass_matrices,
        )
        first_momenta = jax.tree_util.tree_map(lambda value: value[:, 0], momenta)
        continuations = jax.vmap(
            lambda state, momentum, kinetic_energy, integrator_key: (
                _initialize_transition(
                    state,
                    momentum,
                    kinetic_energy,
                    integrator_key,
                    max_num_doublings=doublings,
                )
            )
        )(
            current_states,
            first_momenta,
            kinetic_energies[:, 0],
            integrator_keys[:, 0],
        )
        scalar_shape = (chains, count)
        buffers = _NUTSBuffers(
            position=jax.tree_util.tree_map(
                lambda value: jnp.zeros(
                    (chains, count, *value.shape[1:]),
                    dtype=value.dtype,
                ),
                current_states.position,
            ),
            logdensity=jnp.zeros(scalar_shape, dtype=current_states.logdensity.dtype),
            acceptance_rate=jnp.zeros(
                scalar_shape,
                dtype=current_states.logdensity.dtype,
            ),
            is_divergent=jnp.zeros(scalar_shape, dtype=bool),
            energy=jnp.zeros(scalar_shape, dtype=current_states.logdensity.dtype),
            num_integration_steps=jnp.zeros(
                scalar_shape,
                dtype=jnp.asarray(0).dtype,
            ),
            num_trajectory_expansions=jnp.zeros(
                scalar_shape,
                dtype=jnp.asarray(0).dtype,
            ),
        )
        initial = _SchedulerCarry(
            continuations=continuations,
            completed=jnp.zeros((chains,), dtype=jnp.int32),
            buffers=buffers,
            num_scheduler_steps=jnp.asarray(0, dtype=jnp.int32),
        )

        def has_unfinished_chains(carry):
            return jnp.any(carry.completed < count)

        def advance_chains(carry):
            raw_continuations, raw_emitted, emissions = jax.vmap(
                lambda continuation, step_size, inverse_mass_matrix: _advance_one_quantum(
                    continuation,
                    step_size,
                    inverse_mass_matrix,
                    logdensity_fn=logdensity_fn,
                    max_num_doublings=doublings,
                    divergence_threshold=threshold,
                )
            )(carry.continuations, step_sizes, inverse_mass_matrices)
            active = carry.completed < count
            emitted = raw_emitted & active
            continuations_after_work = jax.vmap(_choose_one)(
                active,
                raw_continuations,
                carry.continuations,
            )
            buffers_after_write = jax.vmap(_write_one)(
                carry.buffers,
                emissions,
                carry.completed,
                emitted,
            )
            completed = carry.completed + emitted.astype(jnp.int32)
            next_indices = jnp.minimum(completed, count - 1)
            next_momenta = jax.vmap(_dynamic_index_tree)(momenta, next_indices)
            next_kinetic_energies = jax.vmap(lambda values, index: values[index])(
                kinetic_energies, next_indices
            )
            next_integrator_keys = jax.vmap(lambda values, index: values[index])(
                integrator_keys, next_indices
            )
            next_continuations = jax.vmap(
                lambda state, momentum, kinetic_energy, integrator_key: (
                    _initialize_transition(
                        state,
                        momentum,
                        kinetic_energy,
                        integrator_key,
                        max_num_doublings=doublings,
                    )
                )
            )(
                emissions.state,
                next_momenta,
                next_kinetic_energies,
                next_integrator_keys,
            )
            needs_next_transition = emitted & (completed < count)
            continuations = jax.vmap(_choose_one)(
                needs_next_transition,
                next_continuations,
                continuations_after_work,
            )
            return _SchedulerCarry(
                continuations=continuations,
                completed=completed,
                buffers=buffers_after_write,
                num_scheduler_steps=carry.num_scheduler_steps + 1,
            )

        result = jax.lax.while_loop(has_unfinished_chains, advance_chains, initial)
        output_metrics = {
            "log_density": result.buffers.logdensity,
            "acceptance_rate": result.buffers.acceptance_rate,
            "divergent": result.buffers.is_divergent,
            "energy": result.buffers.energy,
            "num_integration_steps": result.buffers.num_integration_steps,
            "num_trajectory_expansions": result.buffers.num_trajectory_expansions,
        }
        return (
            result.continuations.current_state,
            result.buffers.position,
            output_metrics,
            InterleavedNUTSStats(result.num_scheduler_steps),
        )

    return jax.jit(run)


def _initialize_transition(
    state,
    momentum,
    kinetic_energy,
    integrator_key,
    *,
    max_num_doublings,
):
    integrator_state = integrators.IntegratorState(
        state.position,
        momentum,
        state.logdensity,
        state.logdensity_grad,
    )
    initial_energy = -state.logdensity + kinetic_energy
    initial_proposal = _Proposal(
        integrator_state,
        initial_energy,
        jnp.zeros_like(initial_energy),
        jnp.full_like(initial_energy, -jnp.inf),
    )
    initial_trajectory = _Trajectory(
        integrator_state,
        integrator_state,
        momentum,
        jnp.asarray(0),
    )
    continuation = _NUTSContinuation(
        current_state=state,
        integrator_key=integrator_key,
        initial_energy=initial_energy,
        expansion_index=jnp.asarray(0),
        global_proposal=initial_proposal,
        global_trajectory=initial_trajectory,
        termination_state=_new_termination_state(
            state.position,
            max_num_doublings=max_num_doublings,
        ),
        local_step=jnp.asarray(0),
        local_proposal=initial_proposal,
        local_trajectory=initial_trajectory,
    )
    return _initialize_subtrajectory(continuation)


def _new_termination_state(position, *, max_num_doublings):
    flat_position, _ = ravel_pytree(position)
    checkpoints = jnp.zeros((max_num_doublings, flat_position.shape[0]))
    zero = jnp.asarray(0, dtype=jnp.int32)
    return _IterativeUTurnState(checkpoints, checkpoints, zero, zero)


def _initialize_subtrajectory(continuation):
    expansion_key = jr.fold_in(
        continuation.integrator_key,
        continuation.expansion_index,
    )
    direction_key, _, _ = jr.split(expansion_key, 3)
    direction = jnp.where(jr.bernoulli(direction_key), 1, -1)
    start_state = jax.lax.cond(
        direction > 0,
        lambda: continuation.global_trajectory.rightmost_state,
        lambda: continuation.global_trajectory.leftmost_state,
    )
    local_proposal = _Proposal(
        start_state,
        continuation.initial_energy,
        jnp.zeros_like(continuation.initial_energy),
        jnp.full_like(continuation.initial_energy, -jnp.inf),
    )
    local_trajectory = _Trajectory(
        start_state,
        start_state,
        start_state.momentum,
        jnp.asarray(0),
    )
    return continuation._replace(
        local_step=jnp.asarray(0),
        local_proposal=local_proposal,
        local_trajectory=local_trajectory,
    )


def _advance_one_quantum(
    continuation,
    step_size,
    inverse_mass_matrix,
    *,
    logdensity_fn,
    max_num_doublings,
    divergence_threshold,
    metric_override=None,
    integrator_override=None,
):
    metric = (
        metrics.default_metric(inverse_mass_matrix)
        if metric_override is None
        else metric_override
    )
    integrator = (
        integrators.velocity_verlet(logdensity_fn, metric.kinetic_energy)
        if integrator_override is None
        else integrator_override
    )
    _, generate_proposal = proposal.proposal_generator(
        trajectory.hmc_energy(metric.kinetic_energy)
    )
    _, update_termination_state, is_criterion_met = termination.iterative_uturn_numpyro(
        metric.check_turning
    )
    expansion_key = jr.fold_in(
        continuation.integrator_key,
        continuation.expansion_index,
    )
    direction_key, trajectory_key, global_proposal_key = jr.split(expansion_key, 3)
    direction = jnp.where(jr.bernoulli(direction_key), 1, -1)
    local_proposal_key = jr.fold_in(trajectory_key, continuation.local_step)
    new_state = integrator(
        continuation.local_trajectory.rightmost_state,
        direction * step_size,
    )
    new_proposal = generate_proposal(continuation.initial_energy, new_state)
    is_divergent = -new_proposal.weight > divergence_threshold

    def initialize_local(_):
        return (
            _Trajectory(
                new_state,
                new_state,
                new_state.momentum,
                jnp.asarray(1),
            ),
            new_proposal,
        )

    def extend_local(_):
        return (
            trajectory.append_to_trajectory(
                continuation.local_trajectory,
                new_state,
            ),
            proposal.progressive_uniform_sampling(
                local_proposal_key,
                continuation.local_proposal,
                new_proposal,
            ),
        )

    local_trajectory, local_proposal = jax.lax.cond(
        continuation.local_step == 0,
        initialize_local,
        extend_local,
        operand=None,
    )
    termination_state = update_termination_state(
        continuation.termination_state,
        local_trajectory.momentum_sum,
        new_state.momentum,
        continuation.local_step,
    )
    is_turning_subtree = is_criterion_met(
        termination_state,
        local_trajectory.momentum_sum,
        new_state.momentum,
    )
    next_local_step = continuation.local_step + 1
    target_steps = jnp.left_shift(
        jnp.asarray(1),
        continuation.expansion_index,
    )
    local_done = (next_local_step >= target_steps) | is_divergent | is_turning_subtree

    def keep_integrating(_):
        updated = continuation._replace(
            local_step=next_local_step,
            local_proposal=local_proposal,
            local_trajectory=local_trajectory,
            termination_state=termination_state,
        )
        return updated, jnp.asarray(False), _placeholder_emission(updated)

    def finish_subtrajectory(_):
        ordered_local_trajectory = jax.lax.cond(
            direction > 0,
            lambda: local_trajectory,
            lambda: _Trajectory(
                local_trajectory.rightmost_state,
                local_trajectory.leftmost_state,
                local_trajectory.momentum_sum,
                local_trajectory.num_states,
            ),
        )

        def accumulate_acceptance(_):
            return _Proposal(
                continuation.global_proposal.state,
                continuation.global_proposal.energy,
                continuation.global_proposal.weight,
                jnp.logaddexp(
                    continuation.global_proposal.sum_log_p_accept,
                    local_proposal.sum_log_p_accept,
                ),
            )

        def sample_global_proposal(_):
            return proposal.progressive_biased_sampling(
                global_proposal_key,
                continuation.global_proposal,
                local_proposal,
            )

        global_proposal = jax.lax.cond(
            is_divergent | is_turning_subtree,
            accumulate_acceptance,
            sample_global_proposal,
            operand=None,
        )
        left_trajectory, right_trajectory = _reorder_trajectories(
            direction,
            continuation.global_trajectory,
            ordered_local_trajectory,
        )
        global_trajectory = trajectory.merge_trajectories(
            left_trajectory,
            right_trajectory,
        )
        left_momentum, _ = ravel_pytree(global_trajectory.leftmost_state.momentum)
        right_momentum, _ = ravel_pytree(global_trajectory.rightmost_state.momentum)
        momentum_sum, _ = ravel_pytree(global_trajectory.momentum_sum)
        is_turning_global = metric.check_turning(
            left_momentum,
            right_momentum,
            momentum_sum,
        )
        next_expansion_index = continuation.expansion_index + 1
        transition_done = (
            is_divergent
            | is_turning_subtree
            | is_turning_global
            | (next_expansion_index >= max_num_doublings)
        )
        updated = continuation._replace(
            expansion_index=next_expansion_index,
            global_proposal=global_proposal,
            global_trajectory=global_trajectory,
            termination_state=termination_state,
        )

        def finish_transition(_):
            sampled_state = global_proposal.state
            state = hmc.HMCState(
                sampled_state.position,
                sampled_state.logdensity,
                sampled_state.logdensity_grad,
            )
            final_continuation = updated._replace(current_state=state)
            emission = _NUTSEmission(
                state=state,
                logdensity=jnp.asarray(state.logdensity),
                acceptance_rate=(
                    jnp.exp(global_proposal.sum_log_p_accept)
                    / global_trajectory.num_states
                ),
                is_divergent=is_divergent,
                energy=global_proposal.energy,
                num_integration_steps=global_trajectory.num_states,
                num_trajectory_expansions=next_expansion_index,
            )
            return final_continuation, jnp.asarray(True), emission

        def continue_transition(_):
            next_continuation = _initialize_subtrajectory(updated)
            return (
                next_continuation,
                jnp.asarray(False),
                _placeholder_emission(next_continuation),
            )

        return jax.lax.cond(
            transition_done,
            finish_transition,
            continue_transition,
            operand=None,
        )

    return jax.lax.cond(
        local_done,
        finish_subtrajectory,
        keep_integrating,
        operand=None,
    )


def _placeholder_emission(continuation):
    zero = jnp.zeros_like(continuation.initial_energy)
    return _NUTSEmission(
        state=continuation.current_state,
        logdensity=continuation.current_state.logdensity,
        acceptance_rate=zero,
        is_divergent=jnp.asarray(False),
        energy=zero,
        num_integration_steps=jnp.asarray(0),
        num_trajectory_expansions=jnp.asarray(0),
    )


def _choose_one(condition, when_true, when_false):
    return jax.lax.cond(
        condition,
        lambda _: when_true,
        lambda _: when_false,
        operand=None,
    )


def _dynamic_index_tree(values, index):
    return jax.tree_util.tree_map(lambda value: value[index], values)


def _write_one(buffers, emission, index, should_write):
    safe_index = jnp.minimum(index, buffers.logdensity.shape[0] - 1)

    def write(_):
        return _NUTSBuffers(
            position=jax.tree_util.tree_map(
                lambda buffer, value: buffer.at[safe_index].set(value),
                buffers.position,
                emission.state.position,
            ),
            logdensity=buffers.logdensity.at[safe_index].set(emission.logdensity),
            acceptance_rate=buffers.acceptance_rate.at[safe_index].set(
                emission.acceptance_rate
            ),
            is_divergent=buffers.is_divergent.at[safe_index].set(emission.is_divergent),
            energy=buffers.energy.at[safe_index].set(emission.energy),
            num_integration_steps=buffers.num_integration_steps.at[safe_index].set(
                emission.num_integration_steps
            ),
            num_trajectory_expansions=buffers.num_trajectory_expansions.at[
                safe_index
            ].set(emission.num_trajectory_expansions),
        )

    return jax.lax.cond(
        should_write,
        write,
        lambda _: buffers,
        operand=None,
    )


__all__ = ["InterleavedNUTSStats", "build_interleaved_nuts_advancer"]
