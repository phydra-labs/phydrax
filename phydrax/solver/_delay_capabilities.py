#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class StochasticDelayInterpolationCapabilities(StrictModule, NonTrainableState):
    interpretation: Literal["ito", "stratonovich"] = eqx.field(static=True)
    strong_order: float = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    replayable: bool = eqx.field(static=True)
    requires_levy_area: bool = eqx.field(static=True)
    geometry_kind: Literal["euclidean", "manifold"] = eqx.field(static=True)
    supported_noise_structures: tuple[str, ...] = eqx.field(static=True)


class AcceptedStochasticDelayInterpolation(StrictModule, NonTrainableState):
    start_time: Array
    end_time: Array
    start_state: Array
    end_state: Array
    midpoint_state: Array
    interpolation_id: str = eqx.field(static=True)

    def evaluate(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time)
        theta = (time_ - self.start_time) / (self.end_time - self.start_time)
        left = self.start_state + 2 * theta * (self.midpoint_state - self.start_state)
        right = self.midpoint_state + (2 * theta - 1) * (
            self.end_state - self.midpoint_state
        )
        return jnp.where((theta <= 0.5)[..., None], left, right)


class AbstractStochasticDelayInterpolation(StrictModule):
    capabilities: StochasticDelayInterpolationCapabilities
    interpolation_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def accepted_step(
        self,
        start_time: Array,
        end_time: Array,
        start_state: Array,
        midpoint_state: Array,
        end_state: Array,
        /,
    ) -> AcceptedStochasticDelayInterpolation:
        raise NotImplementedError


class _TwoHalfStepInterpolation(AbstractStochasticDelayInterpolation, NonTrainableState):
    __strict_abstract__ = True

    def accepted_step(
        self, start_time, end_time, start_state, midpoint_state, end_state, /
    ):
        return AcceptedStochasticDelayInterpolation(
            jnp.asarray(start_time),
            jnp.asarray(end_time),
            jnp.asarray(start_state),
            jnp.asarray(end_state),
            jnp.asarray(midpoint_state),
            self.interpolation_id,
        )


class ItoEulerDelayInterpolation(_TwoHalfStepInterpolation):
    def __init__(self):
        self.capabilities = StochasticDelayInterpolationCapabilities(
            "ito",
            0.5,
            True,
            True,
            False,
            "euclidean",
            ("additive", "diagonal", "general"),
        )
        self.interpolation_id = "stochastic-delay:ito-euler:two-half"


class StratonovichEulerHeunDelayInterpolation(_TwoHalfStepInterpolation):
    def __init__(self):
        self.capabilities = StochasticDelayInterpolationCapabilities(
            "stratonovich",
            0.5,
            True,
            True,
            False,
            "euclidean",
            ("additive", "diagonal", "general"),
        )
        self.interpolation_id = "stochastic-delay:stratonovich-euler-heun:two-half"


class SRKMKDelayInterpolation(_TwoHalfStepInterpolation):
    def __init__(self, *, requires_levy_area: bool = False):
        self.capabilities = StochasticDelayInterpolationCapabilities(
            "stratonovich",
            0.5,
            True,
            True,
            requires_levy_area,
            "manifold",
            ("additive", "diagonal", "general"),
        )
        self.interpolation_id = "stochastic-delay:srkmk:fixed-step"


class AdaptiveStochasticDelayPolicy(StrictModule, NonTrainableState):
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    maximum_accepted_steps: int = eqx.field(static=True)
    failure: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        relative_tolerance: float,
        absolute_tolerance: float,
        minimum_step: float,
        maximum_step: float,
        maximum_attempts: int,
        maximum_accepted_steps: int,
        /,
        *,
        failure: int = -1,
    ):
        values = tuple(
            float(value)
            for value in (
                relative_tolerance,
                absolute_tolerance,
                minimum_step,
                maximum_step,
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0 for value in values)
            or values[2] > values[3]
        ):
            raise ValueError(
                "stochastic delay tolerances/step bounds must be positive, finite, and ordered."
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (maximum_attempts, maximum_accepted_steps)
        ):
            raise ValueError("stochastic delay capacities must be positive integers.")
        (
            self.relative_tolerance,
            self.absolute_tolerance,
            self.minimum_step,
            self.maximum_step,
        ) = values
        self.maximum_attempts = maximum_attempts
        self.maximum_accepted_steps = maximum_accepted_steps
        self.failure = int(failure)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "adaptive-stochastic-delay",
                "rtol": values[0],
                "atol": values[1],
                "minimum_step": values[2],
                "maximum_step": values[3],
                "maximum_attempts": maximum_attempts,
                "maximum_accepted_steps": maximum_accepted_steps,
            }
        )


class StochasticDelayControllerEvidence(StrictModule, NonTrainableState):
    attempt_times: Array
    attempt_steps: Array
    strong_errors: Array
    accepted_attempts: Array
    accepted_times: Array
    accepted_states: Array
    attempt_active: Array
    accepted_active: Array
    attempt_count: Array
    accepted_count: Array
    path_id: str = eqx.field(static=True)
    status: Array
    capacity_exceeded: Array
    policy_id: str = eqx.field(static=True)


def adaptive_stochastic_delay_step_doubling(
    policy: AdaptiveStochasticDelayPolicy,
    interpolation: AbstractStochasticDelayInterpolation,
    initial_time: ArrayLike,
    terminal_time: ArrayLike,
    initial_state: ArrayLike,
    initial_step: ArrayLike,
    increment: Callable[[Array, Array, Key], Array],
    step: Callable[[Array, Array, Array, Array, Any], Array],
    key: Key,
    /,
    *,
    args: Any = None,
    causal_maximum_step: ArrayLike = jnp.inf,
    path_id: str,
) -> tuple[
    Array,
    StochasticDelayControllerEvidence,
    tuple[AcceptedStochasticDelayInterpolation, ...],
]:
    """Brownian-consistent adaptive step doubling with accepted-only history."""

    if not isinstance(policy, AdaptiveStochasticDelayPolicy):
        raise TypeError("policy must be an AdaptiveStochasticDelayPolicy.")
    if not isinstance(interpolation, AbstractStochasticDelayInterpolation):
        raise TypeError(
            "interpolation must be typed AbstractStochasticDelayInterpolation."
        )
    capabilities = interpolation.capabilities
    if (
        not capabilities.causal
        or not capabilities.replayable
        or capabilities.geometry_kind != "euclidean"
    ):
        raise ValueError(
            "adaptive stochastic delay currently requires causal replayable Euclidean interpolation."
        )
    t0 = jnp.asarray(initial_time)
    t1 = jnp.asarray(terminal_time, dtype=t0.dtype)
    state0 = jnp.asarray(initial_state)
    step0 = jnp.minimum(
        jnp.asarray(initial_step, dtype=t0.dtype),
        jnp.asarray(causal_maximum_step, dtype=t0.dtype),
    )
    n = policy.maximum_attempts
    m = policy.maximum_accepted_steps
    attempt_times = jnp.zeros((n,), dtype=t0.dtype)
    attempt_steps = jnp.zeros((n,), dtype=t0.dtype)
    errors = jnp.zeros((n,), dtype=state0.real.dtype)
    accepted_attempts = jnp.zeros((n,), dtype=bool)
    attempt_active = jnp.zeros((n,), dtype=bool)
    accepted_times = jnp.zeros((m + 1,), dtype=t0.dtype).at[0].set(t0)
    accepted_states = (
        jnp.zeros((m + 1,) + state0.shape, dtype=state0.dtype).at[0].set(state0)
    )
    accepted_active = jnp.zeros((m + 1,), dtype=bool).at[0].set(True)

    def body(index, carry):
        (
            time,
            state,
            dt,
            accepted_count,
            finished,
            failed,
            attempt_times_,
            attempt_steps_,
            errors_,
            accepted_attempts_,
            attempt_active_,
            accepted_times_,
            accepted_states_,
            accepted_active_,
        ) = carry
        active = (~finished) & (~failed)
        dt = jnp.minimum(dt, t1 - time)
        midpoint = time + 0.5 * dt
        end = time + dt
        first_increment = increment(time, midpoint, key)
        second_increment = increment(midpoint, end, key)
        full_increment = first_increment + second_increment
        full = step(time, end, state, full_increment, args)
        half = step(time, midpoint, state, first_increment, args)
        two_half = step(midpoint, end, half, second_increment, args)
        scale = policy.absolute_tolerance + policy.relative_tolerance * jnp.maximum(
            jnp.abs(full), jnp.abs(two_half)
        )
        error = jnp.sqrt(jnp.mean(jnp.square(jnp.abs((two_half - full) / scale))))
        finite = jnp.isfinite(error) & jnp.all(jnp.isfinite(two_half))
        accept = active & finite & (error <= 1.0) & (accepted_count < m)
        next_count = accepted_count + accept.astype(jnp.int32)
        safe_slot = jnp.minimum(next_count, m)
        next_time = jnp.where(accept, end, time)
        next_state = jnp.where(accept, two_half, state)
        factor = jnp.where(error > 0, 0.9 * error ** (-0.5), 2.0)
        proposed = jnp.clip(
            dt * jnp.clip(factor, 0.25, 2.0), policy.minimum_step, policy.maximum_step
        )
        proposed = jnp.minimum(proposed, jnp.asarray(causal_maximum_step, dtype=dt.dtype))
        next_finished = finished | (accept & (end >= t1))
        next_failed = failed | (
            active
            & (
                (~finite)
                | ((~accept) & (dt <= policy.minimum_step))
                | (accepted_count >= m)
            )
        )
        return (
            next_time,
            next_state,
            proposed,
            next_count,
            next_finished,
            next_failed,
            attempt_times_.at[index].set(jnp.where(active, time, 0)),
            attempt_steps_.at[index].set(jnp.where(active, dt, 0)),
            errors_.at[index].set(jnp.where(active, error, 0)),
            accepted_attempts_.at[index].set(accept),
            attempt_active_.at[index].set(active),
            accepted_times_.at[safe_slot].set(
                jnp.where(accept, end, accepted_times_[safe_slot])
            ),
            accepted_states_.at[safe_slot].set(
                jnp.where(accept, two_half, accepted_states_[safe_slot])
            ),
            accepted_active_.at[safe_slot].set(accept | accepted_active_[safe_slot]),
        )

    initial = (
        t0,
        state0,
        step0,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        attempt_times,
        attempt_steps,
        errors,
        accepted_attempts,
        attempt_active,
        accepted_times,
        accepted_states,
        accepted_active,
    )
    final = jax.lax.fori_loop(0, n, body, initial)
    (
        time,
        state,
        _,
        accepted_count,
        finished,
        failed,
        attempt_times,
        attempt_steps,
        errors,
        accepted_attempts,
        attempt_active,
        accepted_times,
        accepted_states,
        accepted_active,
    ) = final
    capacity = failed | (~finished)
    evidence = StochasticDelayControllerEvidence(
        attempt_times,
        attempt_steps,
        errors,
        accepted_attempts,
        accepted_times,
        accepted_states,
        attempt_active,
        accepted_active,
        jnp.sum(attempt_active.astype(jnp.int32)),
        accepted_count,
        path_id,
        jnp.where(capacity, policy.failure, 0).astype(jnp.int32),
        capacity,
        policy.policy_id,
    )
    # Fixed-capacity numerical execution is canonical; Python interpolation objects are
    # intentionally omitted from traced output and can be reconstructed from evidence.
    return state, evidence, ()


class ExponentialConvolutionDelay(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    rates: Array
    weights: Array
    initial_moments: Array
    reducer: Callable[[Array, Any], Array]
    memory_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        rates: ArrayLike,
        weights: ArrayLike,
        initial_moments: ArrayLike,
        /,
        *,
        reducer: Callable[[Array, Any], Array] | None = None,
    ):
        rates_ = jnp.asarray(rates)
        weights_ = jnp.asarray(weights)
        moments = jnp.asarray(initial_moments)
        if (
            rates_.ndim != 1
            or rates_.size == 0
            or weights_.shape != rates_.shape
            or moments.shape[0] != rates_.size
        ):
            raise ValueError(
                "exponential convolution rates/weights/moments must share one nonempty rank axis."
            )
        if (
            np.any(np.asarray(rates_) <= 0)
            or np.any(~np.isfinite(np.asarray(rates_)))
            or np.any(~np.isfinite(np.asarray(weights_)))
        ):
            raise ValueError(
                "exponential convolution rates must be positive and all coefficients finite."
            )
        self.name = name
        self.rates = rates_
        self.weights = weights_
        self.initial_moments = moments
        self.reducer = (lambda value, args: value) if reducer is None else reducer
        self.memory_id = canonical_fingerprint(
            {
                "kind": "exponential-convolution-delay",
                "name": name,
                "rank": int(rates_.size),
            }
        )

    def observation(self, moments: ArrayLike, args: Any = None, /) -> Array:
        values = jnp.asarray(moments)
        combined = ein.contract("r,r...->...", self.weights, values)
        return jnp.asarray(self.reducer(combined, args))

    def advance(
        self, moments: ArrayLike, state: ArrayLike, step_size: ArrayLike, /
    ) -> Array:
        moments_ = jnp.asarray(moments)
        state_ = jnp.asarray(state)
        dt = jnp.asarray(step_size, dtype=self.rates.dtype)
        decay = jnp.exp(-self.rates * dt).reshape((self.rates.size,) + (1,) * state_.ndim)
        gain = ((1 - jnp.exp(-self.rates * dt)) / self.rates).reshape(
            (self.rates.size,) + (1,) * state_.ndim
        )
        return decay * moments_ + gain * state_


class CertifiedTruncatedFunctionalDelay(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    functional: Callable[[Any, Any], Array]
    retained_window: float = eqx.field(static=True)
    tail_bound: Callable[[Array, Any], Array]
    tolerance: float = eqx.field(static=True)
    memory_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        functional: Callable[[Any, Any], Array],
        retained_window: float,
        tail_bound: Callable[[Array, Any], Array],
        tolerance: float,
        /,
    ):
        window, tolerance_ = float(retained_window), float(tolerance)
        if not callable(functional) or not callable(tail_bound):
            raise TypeError("functional and tail_bound must be callable.")
        if (
            not np.isfinite(window)
            or window <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ < 0
        ):
            raise ValueError(
                "retained_window/tolerance must be finite and nonnegative with positive window."
            )
        self.name = name
        self.functional = functional
        self.retained_window = window
        self.tail_bound = tail_bound
        self.tolerance = tolerance_
        self.memory_id = canonical_fingerprint(
            {
                "kind": "certified-truncated-functional-delay",
                "name": name,
                "window": window,
                "tolerance": tolerance_,
            }
        )


class InfiniteMemoryEvidence(StrictModule, NonTrainableState):
    exact: Array
    truncated: Array
    realization_dimension: Array
    retained_window: Array
    tail_bound: Array
    tolerance: Array
    valid: Array
    memory_occupancy: Array
    status: Array
    memory_id: str = eqx.field(static=True)


def evaluate_certified_truncated_delay(
    term: CertifiedTruncatedFunctionalDelay,
    time: ArrayLike,
    window: Any,
    /,
    *,
    args: Any = None,
    memory_occupancy: ArrayLike = 0,
) -> tuple[Array, InfiniteMemoryEvidence]:
    if not isinstance(term, CertifiedTruncatedFunctionalDelay):
        raise TypeError("term must be CertifiedTruncatedFunctionalDelay.")
    value = jnp.asarray(term.functional(window, args))
    bound = jnp.asarray(term.tail_bound(jnp.asarray(time), args)).reshape(())
    valid = (
        jnp.all(jnp.isfinite(value))
        & jnp.isfinite(bound)
        & (bound >= 0)
        & (bound <= term.tolerance)
    )
    evidence = InfiniteMemoryEvidence(
        jnp.asarray(False),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(term.retained_window),
        bound,
        jnp.asarray(term.tolerance),
        valid,
        jnp.asarray(memory_occupancy, dtype=jnp.int32),
        jnp.where(valid, 0, -1).astype(jnp.int32),
        term.memory_id,
    )
    return jnp.where(valid, value, jnp.nan), evidence


class BacksolveDelayAdjoint(StrictModule, NonTrainableState):
    maximum_backward_steps: int = eqx.field(static=True)
    checkpoints: int = eqx.field(static=True)
    failure: int = eqx.field(static=True)
    adjoint_id: str = eqx.field(static=True)

    def __init__(
        self, maximum_backward_steps: int, checkpoints: int, /, *, failure: int = -1
    ):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (maximum_backward_steps, checkpoints)
        ):
            raise ValueError("backsolve delay capacities must be positive integers.")
        self.maximum_backward_steps = maximum_backward_steps
        self.checkpoints = checkpoints
        self.failure = int(failure)
        self.adjoint_id = canonical_fingerprint(
            {
                "kind": "backsolve-delay-adjoint",
                "maximum_backward_steps": maximum_backward_steps,
                "checkpoints": checkpoints,
            }
        )


class DelayPrimalTape(StrictModule, NonTrainableState):
    times: Array
    states: Array
    active: Array
    discontinuities: Array
    problem_id: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    interpolation_id: str = eqx.field(static=True)
    tape_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        active: ArrayLike,
        discontinuities: ArrayLike = (),
        /,
        *,
        problem_id: str,
        path_id: str = "deterministic",
        interpolation_id: str = "piecewise-linear",
    ):
        times_ = jnp.asarray(times)
        states_ = jnp.asarray(states)
        active_ = jnp.asarray(active, dtype=bool)
        if (
            times_.ndim != 1
            or states_.ndim == 0
            or states_.shape[0] != times_.size
            or active_.shape != times_.shape
        ):
            raise ValueError("delay primal tape time/state/active axes must align.")
        discontinuities_ = jnp.asarray(discontinuities, dtype=times_.dtype)
        active_host = np.asarray(active_)
        active_indices = np.flatnonzero(active_host)
        semantic_size = 0 if active_indices.size == 0 else int(active_indices[-1]) + 1
        semantic_mask = active_host[:semantic_size]
        self.times = times_
        self.states = states_
        self.active = active_
        self.discontinuities = discontinuities_
        self.problem_id = problem_id
        self.path_id = path_id
        self.interpolation_id = interpolation_id
        self.tape_id = canonical_fingerprint(
            {
                "kind": "delay-primal-tape",
                "problem": problem_id,
                "path": path_id,
                "interpolation": interpolation_id,
                "active_mask": array_tree_fingerprint(semantic_mask),
                "active_times": array_tree_fingerprint(
                    np.asarray(times_)[:semantic_size][semantic_mask]
                ),
                "active_states": array_tree_fingerprint(
                    np.asarray(states_)[:semantic_size][semantic_mask]
                ),
                "discontinuities": array_tree_fingerprint(discontinuities_),
            }
        )

    def evaluate(self, time: ArrayLike, /) -> Array:
        query = jnp.asarray(time)
        count = jnp.sum(self.active.astype(jnp.int32))
        search_times = jnp.where(
            self.active,
            self.times,
            jnp.asarray(jnp.inf, dtype=self.times.dtype),
        )
        index = jnp.clip(
            jnp.searchsorted(search_times, query, side="right") - 1,
            0,
            jnp.maximum(count - 2, 0),
        )
        left, right = self.times[index], self.times[index + 1]
        theta = (query - left) / (right - left)
        interpolated = self.states[index] + theta * (
            self.states[index + 1] - self.states[index]
        )
        return jnp.where(count > 1, interpolated, self.states[0])


class DelayBacksolveEvidence(StrictModule, NonTrainableState):
    backward_active: Array
    advanced_query_covered: Array
    residual_norms: Array
    backward_steps: Array
    tape_checkpoints: Array
    supported_terms: tuple[str, ...] = eqx.field(static=True)
    valid: Array
    status: Array
    tape_id: str = eqx.field(static=True)
    adjoint_id: str = eqx.field(static=True)


def backsolve_delay_adjoint(
    policy: BacksolveDelayAdjoint,
    tape: DelayPrimalTape,
    drift: Callable[[Array, Array, Array, Any], Array],
    delays: Sequence[float],
    terminal_cotangent: ArrayLike,
    /,
    *,
    args: Any = None,
    loss_impulses: ArrayLike | None = None,
) -> tuple[Array, Any, DelayBacksolveEvidence]:
    """Continuous retarded constant-delay adjoint with archived primal coverage."""
    if not isinstance(policy, BacksolveDelayAdjoint) or not isinstance(
        tape, DelayPrimalTape
    ):
        raise TypeError("policy/tape must be BacksolveDelayAdjoint/DelayPrimalTape.")
    delay_values = tuple(float(value) for value in delays)
    if not delay_values or any(
        not np.isfinite(value) or value <= 0 for value in delay_values
    ):
        raise ValueError(
            "backsolve delays must be a nonempty finite positive constant sequence."
        )
    delays_ = jnp.asarray(delay_values, dtype=tape.times.dtype)
    capacity = tape.times.size
    cotangent = jnp.asarray(terminal_cotangent)
    impulses = (
        jnp.zeros_like(tape.states)
        if loss_impulses is None
        else jnp.asarray(loss_impulses)
    )
    if impulses.shape != tape.states.shape:
        raise ValueError("loss_impulses must match tape states.")
    args_gradient = None if args is None else jax.tree.map(jnp.zeros_like, args)
    if capacity == 0:
        empty_active = jnp.zeros((0,), dtype=bool)
        empty_residuals = jnp.zeros((0,), dtype=tape.states.real.dtype)
        evidence = DelayBacksolveEvidence(
            empty_active,
            empty_active,
            empty_residuals,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(policy.checkpoints, dtype=jnp.int32),
            ("constant-point-delay", "static-distributed-quadrature"),
            jnp.asarray(False),
            jnp.asarray(policy.failure, dtype=jnp.int32),
            tape.tape_id,
            policy.adjoint_id,
        )
        failed_args_gradient = (
            None
            if args_gradient is None
            else jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), args_gradient)
        )
        return jnp.full_like(cotangent, jnp.nan), failed_args_gradient, evidence

    active_count = jnp.sum(tape.active.astype(jnp.int32))
    expected_active = jnp.arange(capacity, dtype=jnp.int32) < active_count
    active_prefix_valid = (active_count > 0) & jnp.all(tape.active == expected_active)
    active_state_mask = tape.active.reshape((capacity,) + (1,) * (tape.states.ndim - 1))
    active_primals_finite = jnp.all(
        jnp.isfinite(jnp.where(tape.active, tape.times, 0))
    ) & jnp.all(jnp.isfinite(jnp.where(active_state_mask, tape.states, 0)))
    within_capacity = active_count - 1 <= policy.maximum_backward_steps
    execution_valid = active_prefix_valid & active_primals_finite & within_capacity
    terminal_index = jnp.maximum(active_count - 1, 0)
    terminal_time = tape.times[terminal_index]
    search_times = jnp.where(
        tape.active,
        tape.times,
        jnp.asarray(jnp.inf, dtype=tape.times.dtype),
    )
    terminal_value = cotangent + impulses[terminal_index]
    lambdas = jnp.zeros_like(tape.states).at[terminal_index].set(terminal_value)
    residuals = jnp.zeros((capacity - 1,), dtype=tape.states.real.dtype)
    active = jnp.zeros((capacity - 1,), dtype=bool)
    covered = jnp.zeros((capacity - 1,), dtype=bool)

    def body(reverse_index, carry):
        index = capacity - 2 - reverse_index
        interval_active = execution_valid & (index < active_count - 1)

        def active_body(active_carry):
            lambdas_, residuals_, active_, covered_, args_gradient_ = active_carry
            time = tape.times[index]
            dt = tape.times[index + 1] - time
            state = tape.states[index]
            delayed = jax.vmap(lambda delay: tape.evaluate(time - delay))(delays_)
            future_lambda = lambdas_[index + 1]
            if args is None:
                _, pullback = jax.vjp(
                    lambda y, z: drift(time, y, z, None), state, delayed
                )
                state_action, _ = pullback(future_lambda)
                next_args_gradient = None
            else:
                _, pullback = jax.vjp(
                    lambda y, z, a: drift(time, y, z, a), state, delayed, args
                )
                state_action, _, parameter_action = pullback(future_lambda)
                next_args_gradient = jax.tree.map(
                    lambda accumulated, value: accumulated + dt * value,
                    args_gradient_,
                    parameter_action,
                )
            advanced = jnp.zeros_like(state)
            advanced_valid = jnp.asarray(True)
            for delay_index in range(delays_.size):
                future_time = time + delays_[delay_index]
                in_domain = future_time <= terminal_time
                bounded_future_time = jnp.minimum(future_time, terminal_time)
                future_state = tape.evaluate(bounded_future_time)
                future_delayed = jax.vmap(
                    lambda delay: tape.evaluate(bounded_future_time - delay)
                )(delays_)
                future_index = jnp.clip(
                    jnp.searchsorted(search_times, bounded_future_time, side="right") - 1,
                    jnp.minimum(index + 1, active_count - 2),
                    active_count - 2,
                )
                left = tape.times[future_index]
                right = tape.times[future_index + 1]
                theta = (bounded_future_time - left) / (right - left)
                future_adjoint = lambdas_[future_index] + theta * (
                    lambdas_[future_index + 1] - lambdas_[future_index]
                )
                _, delayed_pullback = jax.vjp(
                    lambda z: drift(future_time, future_state, z, args),
                    future_delayed,
                )
                delayed_action = delayed_pullback(future_adjoint)[0][delay_index]
                advanced = advanced + jnp.where(in_domain, delayed_action, 0)
                advanced_valid = advanced_valid & ((~in_domain) | (future_time >= time))
            derivative = state_action + advanced
            value = future_lambda + dt * derivative + impulses[index]
            valid = advanced_valid & jnp.all(jnp.isfinite(value)) & (dt > 0)
            value = jnp.where(valid, value, jnp.nan)
            return (
                lambdas_.at[index].set(value),
                residuals_.at[index].set(
                    jnp.sqrt(
                        jnp.mean(
                            jnp.square(
                                jnp.abs(
                                    value
                                    - future_lambda
                                    - dt * derivative
                                    - impulses[index]
                                )
                            )
                        )
                    )
                ),
                active_.at[index].set(True),
                covered_.at[index].set(advanced_valid),
                next_args_gradient,
            )

        return jax.lax.cond(
            interval_active,
            active_body,
            lambda inactive_carry: inactive_carry,
            carry,
        )

    lambdas, residuals, active, covered, args_gradient = jax.lax.fori_loop(
        0, capacity - 1, body, (lambdas, residuals, active, covered, args_gradient)
    )
    valid = (
        execution_valid
        & jnp.all(jnp.isfinite(terminal_value))
        & jnp.all((~active) | covered)
        & jnp.all(jnp.isfinite(jnp.where(active, residuals, 0)))
    )
    evidence = DelayBacksolveEvidence(
        active,
        covered,
        residuals,
        jnp.sum(active.astype(jnp.int32)),
        jnp.asarray(policy.checkpoints, dtype=jnp.int32),
        ("constant-point-delay", "static-distributed-quadrature"),
        valid,
        jnp.where(valid, 0, policy.failure).astype(jnp.int32),
        tape.tape_id,
        policy.adjoint_id,
    )
    args_gradient = (
        None
        if args_gradient is None
        else jax.tree.map(
            lambda leaf: jnp.where(valid, leaf, jnp.full_like(leaf, jnp.nan)),
            args_gradient,
        )
    )
    return jnp.where(valid, lambdas[0], jnp.nan), args_gradient, evidence


__all__ = [
    "AbstractStochasticDelayInterpolation",
    "AcceptedStochasticDelayInterpolation",
    "AdaptiveStochasticDelayPolicy",
    "BacksolveDelayAdjoint",
    "CertifiedTruncatedFunctionalDelay",
    "DelayBacksolveEvidence",
    "DelayPrimalTape",
    "ExponentialConvolutionDelay",
    "InfiniteMemoryEvidence",
    "ItoEulerDelayInterpolation",
    "SRKMKDelayInterpolation",
    "StochasticDelayControllerEvidence",
    "StochasticDelayInterpolationCapabilities",
    "StratonovichEulerHeunDelayInterpolation",
    "adaptive_stochastic_delay_step_doubling",
    "backsolve_delay_adjoint",
    "evaluate_certified_truncated_delay",
]
