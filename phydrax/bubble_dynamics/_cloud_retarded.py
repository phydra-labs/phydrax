#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Retarded (finite sound speed) neighbour coupling of a fixed-position cloud.

The instantaneous near field `Σ_j Q̇_j(t)/d_ij` is replaced by the retarded
source `Σ_j Q̇_j(t − τ_ij)/d_ij` with `τ_ij = d_ij/c` and `Q̇ = R²R̈ + 2RṘ²`,
so every bubble obeys `a_i R̈_i = f_i − Σ_{j≠i} Q̇_j(t − τ_ij)/d_ij`. Delayed
accelerations make this a neutral delay equation. It is solved in the
derivative-delay form of the native delay substrate (`DerivativeDelay` terms,
one constant lag per unordered pair, `solve_diffrax_delay`).
`NeutralDelayProblem`'s transformed-state form `d(y − N(y_t))/dt = F` needs a
state-independent neutral coefficient, whereas here the coefficient `1/a_i`
of the delayed accelerations depends on the current state, so the
derivative-delay form is the exact representation.

The neutral part `R̈_i ≈ −Σ_j (R_j²/(a_i d_ij)) R̈_j(t − τ_ij)` is stable for every
delay iff `ρ(W) < 1` with the symmetric `W = D^{-1/2} G D^{-1/2}`; the plan
refuses an initial `ρ(W) ≥ 1` and a native event stops the solve with
`NEUTRAL_UNSTABLE` if it is reached. The prehistory is the initial state held
constant; the plan requires a derivative-compatible start (bubbles at rest in
equilibrium with `p_d(0) = 0`), so no neutral derivative discontinuity is
propagated along the combinatorial set of pair-lag sums.
"""

from __future__ import annotations

from typing import Protocol

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from ..solver import (
    ConstantDelay,
    DelayDifferentialProblem,
    DelayValues,
    DerivativeDelay,
    solve_diffrax_delay,
)
from ._cloud import BubbleCloudState, cloud_radius, PreparedBubbleCloud
from ._cloud_integration import (
    cloud_emission,
    cloud_event,
    CloudContext,
    CloudIntegration,
    native_solver,
    saved_maximum_radius,
    terminal_status,
)
from ._status import BubbleDynamicsStatus


class _DenseDelayOutput(Protocol):
    """Native dense delay output: values, derivatives and the reached final time."""

    @property
    def final_time(self) -> Array: ...

    def evaluate(self, time: Array, /, *, left: bool = True) -> Array: ...

    def derivative(self, time: Array, /, *, left: bool = True) -> Array: ...


def pair_delays(prepared: PreparedBubbleCloud, /) -> Array:
    """Propagation delay `d_ij/c` (s) of every unordered bubble pair."""
    position = prepared.initial_state.position
    separation = position[prepared.pair_first] - position[prepared.pair_second]
    return jnp.sqrt(jnp.sum(separation**2, axis=-1)) / prepared.plan.sound_speed()


def _neighbor_field(
    prepared: PreparedBubbleCloud,
    delayed: BubbleCloudState,
    delayed_rate: BubbleCloudState,
    /,
) -> tuple[Array, Array]:
    """Retarded potential `Σ_j Q̇_j(t − τ_ij)/d_ij` and its gradient at every bubble.

    `delayed` and `delayed_rate` carry one cloud state (and physical time
    derivative) per pair lag along their leading axis.
    """
    first = prepared.pair_first
    second = prepared.pair_second
    radius = jax.vmap(cloud_radius)(delayed)
    velocity = jnp.concatenate(
        [member.wall_velocity for member in delayed.groups], axis=1
    )
    acceleration = jnp.concatenate(
        [member.wall_velocity for member in delayed_rate.groups], axis=1
    )
    flux_rate = 2.0 * radius * velocity**2 + radius**2 * acceleration
    lag = jnp.arange(first.shape[0])
    position = prepared.initial_state.position
    separation = position[first] - position[second]
    distance = jnp.sqrt(jnp.sum(separation**2, axis=-1))
    from_second = flux_rate[lag, second] / distance
    from_first = flux_rate[lag, first] / distance
    count = position.shape[0]
    potential = (
        jnp.zeros((count,), dtype=position.dtype)
        .at[first]
        .add(from_second)
        .at[second]
        .add(from_first)
    )
    direction = separation / (distance**2)[:, None]
    gradient = (
        jnp.zeros_like(position)
        .at[first]
        .add(-from_second[:, None] * direction)
        .at[second]
        .add(from_first[:, None] * direction)
    )
    return potential, gradient


def _delay_terms(delays: Array, /) -> tuple[ConstantDelay | DerivativeDelay, ...]:
    terms: list[ConstantDelay | DerivativeDelay] = []
    for pair in range(delays.shape[0]):
        terms.append(ConstantDelay(f"state-{pair}", delays[pair]))
        terms.append(
            DerivativeDelay(
                f"rate-{pair}", ConstantDelay(f"rate-lag-{pair}", delays[pair])
            )
        )
    return tuple(terms)


class _RetardedDrift(StrictModule):
    """Delay vector field of the retarded cloud in nondimensional form."""

    context: CloudContext
    pair_count: int = eqx.field(static=True)

    def __init__(self, context: CloudContext, pair_count: int, /) -> None:
        self.context = context
        self.pair_count = pair_count

    def __call__(
        self, time: Array, flat: Array, memory: DelayValues, args: None, /
    ) -> Array:
        del args
        context = self.context
        prepared = context.prepared
        values = jnp.stack([memory[f"state-{pair}"] for pair in range(self.pair_count)])
        rates = jnp.stack([memory[f"rate-{pair}"] for pair in range(self.pair_count)])
        potential, gradient = _retarded_field(context, values, rates)
        state, _ = context.physical(flat)
        derivative = prepared.retarded_rates(
            state, time * prepared.time_scale, potential, gradient
        )
        return context.flatten(derivative)


def _retarded_field(
    context: CloudContext, values: Array, rates: Array, /
) -> tuple[Array, Array]:
    """Neighbour field from one delayed flat state and flat derivative per pair lag."""
    prepared = context.prepared
    states, _ = jax.vmap(context.physical)(values)
    derivatives, _ = jax.vmap(context.physical)(rates)
    physical_rates = jax.tree.map(lambda leaf: leaf / prepared.time_scale, derivatives)
    return _neighbor_field(prepared, states, physical_rates)


def integrate_retarded(context: CloudContext, flat: Array, /) -> CloudIntegration:
    """One native neutral delay solve of the retarded cloud."""
    prepared = context.prepared
    plan = prepared.plan
    time_scale = prepared.time_scale
    delays = pair_delays(prepared) / time_scale
    pair_count = delays.shape[0]
    drift = _RetardedDrift(context, pair_count)
    save_times = plan.save_times / time_scale
    problem = DelayDifferentialProblem(
        drift,
        lambda time, args: flat,
        _delay_terms(delays),
        t0=jnp.zeros(()),
        t1=save_times[-1],
        history_derivative=lambda time, args: jnp.zeros_like(flat),
        problem_id=f"{plan.plan_id}:retarded-cloud",
    )
    event, kinds = cloud_event(context)
    solution = solve_diffrax_delay(
        problem,
        save_times=save_times,
        solver=native_solver(plan.integrator),
        event=event,
        rtol=plan.relative_tolerance,
        atol=plan.absolute_tolerance,
        dense=True,
        history_mode="full",
        max_steps=plan.maximum_steps,
        throw=False,
    )
    stats = solution.stats
    exhausted = solution.backend_result == dfx.RESULTS.max_steps_reached
    event_terminated = solution.backend_result == dfx.RESULTS.event_occurred
    solver_ok = (solution.backend_result == dfx.RESULTS.successful) | event_terminated
    status, kind = terminal_status(
        solver_ok, exhausted, event_terminated, solution.event_mask, kinds
    )
    history_exhausted = jnp.asarray(stats["history_capacity_exhausted"], dtype=jnp.bool_)
    status = jnp.where(
        history_exhausted, int(BubbleDynamicsStatus.HISTORY_CAPACITY), status
    ).astype(jnp.int32)
    if solution.interpolation is None:
        raise ValueError("The retarded cloud requires native dense delay output.")
    interpolation: _DenseDelayOutput = solution.interpolation
    end_time = jnp.asarray(interpolation.final_time).reshape(())
    covered = solution.valid & (save_times <= end_time)
    end_state = interpolation.evaluate(end_time)
    saved = jnp.where(covered[:, None], solution.states, end_state[None, :])
    row_times = jnp.concatenate((save_times, end_time[None]))
    potential, gradient = _row_fields(context, interpolation, flat, row_times, delays)
    maximum_radius = saved_maximum_radius(context, saved, covered, end_state)
    return CloudIntegration(
        end_time,
        end_state,
        saved,
        covered,
        status,
        kind,
        jnp.asarray(stats["num_accepted_steps"], dtype=jnp.int32),
        jnp.asarray(stats["num_rejected_steps"], dtype=jnp.int32),
        solver_ok & ~history_exhausted,
        cloud_emission(context, interpolation.evaluate, end_time, maximum_radius),
        potential,
        gradient,
        jnp.asarray(stats["num_accepted_steps"], dtype=jnp.int32),
    )


def _row_fields(
    context: CloudContext,
    interpolation: _DenseDelayOutput,
    flat: Array,
    row_times: Array,
    delays: Array,
    /,
) -> tuple[Array, Array]:
    """Retarded neighbour field at every saved row from the dense delay output.

    Lags reaching before `t = 0` see the constant prehistory (zero derivative).
    """

    def row(time: Array) -> tuple[Array, Array]:
        query = time - delays
        before = query < 0.0
        clipped = jnp.maximum(query, 0.0)
        values = jnp.where(
            before[:, None], flat[None, :], interpolation.evaluate(clipped)
        )
        rates = jnp.where(before[:, None], 0.0, interpolation.derivative(clipped))
        return _retarded_field(context, values, rates)

    return jax.lax.map(row, row_times)


__all__ = ["integrate_retarded", "pair_delays"]
