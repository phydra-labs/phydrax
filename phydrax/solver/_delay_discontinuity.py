#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import comb
from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._delay import StateDependentDelay


class DynamicDiscontinuityState(eqx.Module):
    """Finite-capacity state for forward discontinuity propagation."""

    times: Array
    generations: Array
    landed: Array
    processed: Array
    count: Array
    bracket_active: Array
    bracket_source: Array
    bracket_delay: Array
    bracket_low_time: Array
    bracket_low_value: Array
    bracket_high_time: Array
    bracket_high_value: Array
    bracket_iterations: Array
    root_times: Array
    num_roots: Array
    num_restarts: Array


class DynamicControllerState(eqx.Module):
    inner_state: Any
    discontinuities: DynamicDiscontinuityState


def constant_discontinuity_schedule(
    delays: Array,
    initial_discontinuities: Array,
    /,
    *,
    depth: int,
    max_discontinuities: int,
) -> tuple[Array, Array]:
    """Return the ordered additive constant-delay schedule and its generations."""
    num_delays = int(delays.size)
    num_sources = int(initial_discontinuities.size)
    combinations = comb(depth + num_delays, num_delays)
    candidate_count = num_sources * combinations
    if candidate_count > max_discontinuities:
        raise ValueError(
            "Delay discontinuity schedule exceeds max_discontinuities; "
            f"requires {candidate_count}, limit is {max_discontinuities}."
        )
    if num_sources == 0:
        return (
            jnp.empty((0,), dtype=delays.dtype),
            jnp.empty((0,), dtype=jnp.int32),
        )
    multi_indices = tuple(
        values
        for values in product(range(depth + 1), repeat=num_delays)
        if sum(values) <= depth
    )
    index_matrix = jnp.asarray(multi_indices, dtype=delays.dtype)
    offsets = index_matrix @ delays
    candidates = initial_discontinuities[:, None] + offsets[None, :]
    generations = jnp.broadcast_to(
        jnp.sum(jnp.asarray(multi_indices), axis=1, dtype=jnp.int32)[None, :],
        candidates.shape,
    )
    flat_candidates = candidates.reshape((-1,))
    flat_generations = generations.reshape((-1,))
    order = jnp.argsort(flat_candidates)
    return flat_candidates[order], flat_generations[order]


class StateDependentDiscontinuityTracker(eqx.Module):
    """Certified monotone and sign-isolated nonmonotone root tracker."""

    delays: tuple[StateDependentDelay, ...]
    constant_delays: Array
    initial_times: Array
    initial_generations: Array
    horizon: Array
    root_rtol: Array
    root_atol: Array
    depth: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    initial_count: int = eqx.field(static=True)
    max_root_iterations: int = eqx.field(static=True)

    maximum_isolation_step: Array

    def __init__(
        self,
        delays: tuple[StateDependentDelay, ...],
        constant_delays: Array,
        initial_times: Array,
        initial_generations: Array,
        horizon: Array,
        /,
        *,
        depth: int,
        capacity: int,
        root_rtol: float,
        root_atol: float,
        max_root_iterations: int,
    ):
        if not delays:
            raise ValueError("A state-dependent tracker requires at least one delay.")
        if depth <= 0:
            raise ValueError("Dynamic discontinuity depth must be positive.")
        if initial_times.ndim != 1 or initial_generations.shape != initial_times.shape:
            raise ValueError("Initial discontinuity times and generations must align.")
        initial_count = int(initial_times.size)
        if initial_count > capacity:
            raise ValueError(
                "Initial discontinuity schedule exceeds the static root capacity."
            )
        if root_rtol < 0.0 or root_atol <= 0.0:
            raise ValueError("root_rtol must be nonnegative and root_atol positive.")
        if max_root_iterations <= 0:
            raise ValueError("max_root_iterations must be positive.")
        padding = capacity - initial_count
        order = jnp.argsort(initial_times)
        ordered_times = initial_times[order]
        ordered_generations = initial_generations[order].astype(jnp.int32)
        if initial_count:

            def reduce_generation(carry, item):
                previous_time, minimum_generation = carry
                item_time, item_generation = item
                same_root = item_time == previous_time
                minimum_generation = jnp.where(
                    same_root,
                    jnp.minimum(minimum_generation, item_generation),
                    item_generation,
                )
                return (item_time, minimum_generation), minimum_generation

            _, grouped_generations = jax.lax.scan(
                reduce_generation,
                (ordered_times[0], ordered_generations[0]),
                (ordered_times, ordered_generations),
            )
            last_in_group = jnp.concatenate(
                (
                    ordered_times[:-1] != ordered_times[1:],
                    jnp.asarray([True]),
                )
            )
            keep = last_in_group & jnp.isfinite(ordered_times)
            compact_times = jnp.where(keep, ordered_times, jnp.inf)
            compact_generations = jnp.where(
                keep,
                grouped_generations,
                jnp.asarray(depth, dtype=jnp.int32),
            )
            compact_order = jnp.argsort(compact_times)
            ordered_times = compact_times[compact_order]
            ordered_generations = compact_generations[compact_order]
        self.delays = delays
        self.constant_delays = jnp.asarray(constant_delays)
        self.initial_times = jnp.pad(
            ordered_times,
            (0, padding),
            constant_values=jnp.inf,
        )
        self.initial_generations = jnp.pad(
            ordered_generations,
            (0, padding),
            constant_values=depth,
        )
        self.horizon = jnp.asarray(horizon, dtype=initial_times.dtype)
        self.root_rtol = jnp.asarray(root_rtol, dtype=initial_times.dtype)
        self.root_atol = jnp.asarray(root_atol, dtype=initial_times.dtype)
        self.depth = depth
        self.capacity = capacity
        self.initial_count = initial_count
        self.max_root_iterations = max_root_iterations
        isolation_steps = []
        for delay in delays:
            if not delay.monotone_argument:
                assert delay.root_isolation_step is not None
                isolation_steps.append(delay.root_isolation_step)
        self.maximum_isolation_step = (
            jnp.min(jnp.stack(isolation_steps))
            if isolation_steps
            else jnp.asarray(jnp.inf, dtype=initial_times.dtype)
        )

    def initial_state(
        self,
        time: Array,
        value: Array,
        args: Any,
        /,
    ) -> DynamicDiscontinuityState:
        tolerance = self._time_tolerance(time)
        finite_count = jnp.sum(jnp.isfinite(self.initial_times), dtype=jnp.int32)
        active = jnp.arange(self.capacity) < finite_count
        landed = active & (self.initial_times <= time + tolerance)
        state = DynamicDiscontinuityState(
            times=self.initial_times,
            generations=self.initial_generations,
            landed=landed,
            processed=jnp.zeros(
                (self.capacity, len(self.delays)),
                dtype=bool,
            ),
            count=finite_count,
            bracket_active=jnp.asarray(False),
            bracket_source=jnp.asarray(0, dtype=jnp.int32),
            bracket_delay=jnp.asarray(0, dtype=jnp.int32),
            bracket_low_time=time,
            bracket_low_value=jnp.zeros(
                (self.capacity, len(self.delays)), dtype=time.dtype
            ),
            bracket_high_time=time,
            bracket_high_value=jnp.zeros(
                (self.capacity, len(self.delays)), dtype=time.dtype
            ),
            bracket_iterations=jnp.asarray(0, dtype=jnp.int32),
            root_times=jnp.full((self.capacity,), jnp.inf, dtype=time.dtype),
            num_roots=jnp.asarray(0, dtype=jnp.int32),
            num_restarts=jnp.asarray(0, dtype=jnp.int32),
        )
        for index in range(self.initial_count):
            source_time = self.initial_times[index]
            source_generation = self.initial_generations[index]
            state = jax.lax.cond(
                landed[index] & jnp.isfinite(source_time),
                lambda current: self._append_constant_children(
                    current,
                    source_time,
                    source_generation,
                ),
                lambda current: current,
                state,
            )
        roots, root_active = self._values(state, time, value, args)
        monotone = jnp.asarray(tuple(delay.monotone_argument for delay in self.delays))[
            None, :
        ]
        already_propagated = (
            root_active & monotone & (roots >= -self._time_tolerance(time))
        )
        return eqx.tree_at(
            lambda item: item.processed,
            state,
            state.processed | already_propagated,
        )

    def _time_tolerance(self, time: Array, /) -> Array:
        return self.root_atol + self.root_rtol * jnp.abs(time)

    def _values(
        self,
        state: DynamicDiscontinuityState,
        time: Array,
        value: Array,
        args: Any,
        /,
    ) -> tuple[Array, Array]:
        delayed_arguments = jnp.stack(
            tuple(time - delay.value(time, value, args) for delay in self.delays)
        )
        roots = delayed_arguments[None, :] - state.times[:, None]
        active_sources = jnp.arange(self.capacity) < state.count
        monotone = jnp.asarray(tuple(delay.monotone_argument for delay in self.delays))[
            None, :
        ]
        active = (
            active_sources[:, None]
            & (state.generations[:, None] < self.depth)
            & (~state.processed | ~monotone)
        )
        return roots, active

    def _next_pending(
        self,
        state: DynamicDiscontinuityState,
        start: Array,
        /,
    ) -> Array:
        active = jnp.arange(self.capacity) < state.count
        tolerance = self._time_tolerance(start)
        pending = active & ~state.landed & (state.times > start + tolerance)
        return jnp.min(jnp.where(pending, state.times, jnp.inf))

    def cap_next(
        self,
        state: DynamicDiscontinuityState,
        start: Array,
        proposed_end: ArrayLike,
        /,
    ) -> Array:
        pending = jax.lax.stop_gradient(self._next_pending(state, start))
        isolation_end = jax.lax.stop_gradient(start + self.maximum_isolation_step)
        return jax.lax.stop_gradient(
            jnp.minimum(jnp.minimum(proposed_end, pending), isolation_end)
        )

    def _sort(self, state: DynamicDiscontinuityState, /) -> DynamicDiscontinuityState:
        order = jnp.argsort(state.times)
        return eqx.tree_at(
            lambda item: (
                item.times,
                item.generations,
                item.landed,
                item.processed,
            ),
            state,
            (
                state.times[order],
                state.generations[order],
                state.landed[order],
                state.processed[order],
            ),
        )

    def _insert_source(
        self,
        state: DynamicDiscontinuityState,
        time: Array,
        generation: Array,
        landed: Array,
        /,
    ) -> tuple[DynamicDiscontinuityState, Array]:
        indices = jnp.arange(self.capacity)
        active = indices < state.count
        tolerance = self._time_tolerance(time)
        matches = active & (jnp.abs(state.times - time) <= tolerance)
        exists = jnp.any(matches)
        existing_index = jnp.argmax(matches).astype(jnp.int32)
        checked_time = eqx.error_if(
            time,
            ~exists & (state.count >= self.capacity),
            "Dynamic delay discontinuity roots exceed max_discontinuities.",
        )
        insert_index = jnp.minimum(state.count, self.capacity - 1)
        index = jnp.where(exists, existing_index, insert_index)
        old_generation = state.generations[index]
        old_landed = state.landed[index]
        new_generation = jnp.where(
            exists,
            jnp.minimum(old_generation, generation),
            generation,
        )
        new_landed = jnp.where(exists, old_landed | landed, landed)
        times = state.times.at[index].set(
            jnp.where(exists, state.times[index], checked_time)
        )
        generations = state.generations.at[index].set(new_generation)
        landed_values = state.landed.at[index].set(new_landed)
        processed = state.processed.at[index].set(
            jnp.where(
                exists,
                state.processed[index],
                jnp.zeros((len(self.delays),), dtype=bool),
            )
        )
        count = state.count + (~exists).astype(jnp.int32)
        updated = eqx.tree_at(
            lambda item: (
                item.times,
                item.generations,
                item.landed,
                item.processed,
                item.count,
            ),
            state,
            (times, generations, landed_values, processed, count),
        )
        return self._sort(updated), ~exists

    def _append_constant_children(
        self,
        state: DynamicDiscontinuityState,
        time: Array,
        generation: Array,
        /,
    ) -> DynamicDiscontinuityState:
        child_generation = generation + 1
        for delay in tuple(self.constant_delays):
            child_time = time + delay

            def add_child(current):
                return self._insert_source(
                    current,
                    child_time,
                    child_generation,
                    jnp.asarray(False),
                )[0]

            state = jax.lax.cond(
                (child_generation <= self.depth)
                & jnp.isfinite(child_time)
                & (child_time < self.horizon),
                add_child,
                lambda current: current,
                state,
            )
        return state

    def mark_pending_landed(
        self,
        state: DynamicDiscontinuityState,
        time: Array,
        /,
    ) -> tuple[DynamicDiscontinuityState, Array]:
        active = jnp.arange(self.capacity) < state.count
        tolerance = self._time_tolerance(time)
        matches = active & ~state.landed & (jnp.abs(state.times - time) <= tolerance)
        landed_any = jnp.any(matches)
        generation = jnp.min(
            jnp.where(matches, state.generations, jnp.asarray(self.depth, jnp.int32))
        )
        landed = state.landed | matches
        state = eqx.tree_at(lambda item: item.landed, state, landed)
        state = jax.lax.cond(
            landed_any,
            lambda current: self._append_constant_children(
                current,
                time,
                generation,
            ),
            lambda current: current,
            state,
        )
        return state, landed_any

    def _crossings(
        self,
        low_values: Array,
        high_values: Array,
        active: Array,
        /,
    ) -> Array:
        forward = (low_values < 0.0) & (high_values >= 0.0)
        reverse = (low_values > 0.0) & (high_values <= 0.0)
        monotone = jnp.asarray(tuple(delay.monotone_argument for delay in self.delays))[
            None, :
        ]
        return active & jnp.where(monotone, forward, forward | reverse)

    def start_bracket(
        self,
        state: DynamicDiscontinuityState,
        t0: Array,
        t1: Array,
        y0: Array,
        y1: Array,
        args: Any,
        /,
    ) -> tuple[DynamicDiscontinuityState, Array, Array]:
        values0, active0 = self._values(state, t0, y0, args)
        values1, active1 = self._values(state, t1, y1, args)
        active = active0 & active1
        crossings = self._crossings(values0, values1, active)
        has_crossing = jnp.any(crossings)
        flat_index = jnp.argmax(crossings.reshape((-1,))).astype(jnp.int32)
        delay_count = len(self.delays)
        source_index = flat_index // delay_count
        delay_index = flat_index % delay_count
        state = eqx.tree_at(
            lambda item: (
                item.bracket_active,
                item.bracket_source,
                item.bracket_delay,
                item.bracket_low_time,
                item.bracket_low_value,
                item.bracket_high_time,
                item.bracket_high_value,
                item.bracket_iterations,
            ),
            state,
            (
                has_crossing,
                source_index,
                delay_index,
                t0,
                values0,
                t1,
                values1,
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        return state, 0.5 * (t0 + t1), has_crossing

    def refine_bracket(
        self,
        state: DynamicDiscontinuityState,
        time: Array,
        value: Array,
        args: Any,
        /,
    ) -> tuple[DynamicDiscontinuityState, Array, Array, Array]:
        values, active = self._values(state, time, value, args)
        left_crossings = self._crossings(
            state.bracket_low_value,
            values,
            active,
        )
        crosses_left = jnp.any(left_crossings)
        flat_index = jnp.argmax(left_crossings.reshape((-1,))).astype(jnp.int32)
        source_index = jnp.where(
            crosses_left,
            flat_index // len(self.delays),
            state.bracket_source,
        )
        delay_index = jnp.where(
            crosses_left,
            flat_index % len(self.delays),
            state.bracket_delay,
        )
        low_time = jnp.where(crosses_left, state.bracket_low_time, time)
        low_values = jnp.where(
            crosses_left,
            state.bracket_low_value,
            values,
        )
        high_time = jnp.where(crosses_left, time, state.bracket_high_time)
        high_values = jnp.where(
            crosses_left,
            values,
            state.bracket_high_value,
        )
        iterations = state.bracket_iterations + 1
        checked_high = eqx.error_if(
            high_time,
            iterations > self.max_root_iterations,
            "State-dependent discontinuity root finder exceeded max_root_iterations.",
        )
        converged = (checked_high - low_time) <= self._time_tolerance(checked_high)
        accept_root = crosses_left & converged
        next_time = jnp.where(
            converged,
            checked_high,
            0.5 * (low_time + checked_high),
        )
        state = eqx.tree_at(
            lambda item: (
                item.bracket_source,
                item.bracket_delay,
                item.bracket_low_time,
                item.bracket_low_value,
                item.bracket_high_time,
                item.bracket_high_value,
                item.bracket_iterations,
            ),
            state,
            (
                source_index,
                delay_index,
                low_time,
                low_values,
                checked_high,
                high_values,
                iterations,
            ),
        )
        return state, next_time, accept_root, ~crosses_left

    def accept_root(
        self,
        state: DynamicDiscontinuityState,
        time: Array,
        value: Array,
        args: Any,
        /,
    ) -> DynamicDiscontinuityState:
        _, active = self._values(state, time, value, args)
        simultaneous = self._crossings(
            state.bracket_low_value,
            state.bracket_high_value,
            active,
        )
        simultaneous = simultaneous.at[
            state.bracket_source,
            state.bracket_delay,
        ].set(True)
        monotone = jnp.asarray(tuple(delay.monotone_argument for delay in self.delays))[
            None, :
        ]
        processed = state.processed | (simultaneous & monotone)
        parent_generations = jnp.broadcast_to(
            state.generations[:, None],
            simultaneous.shape,
        )
        generation = (
            jnp.min(
                jnp.where(
                    simultaneous,
                    parent_generations,
                    jnp.asarray(self.depth, dtype=jnp.int32),
                )
            )
            + 1
        )
        checked_time = eqx.error_if(
            time,
            state.num_roots >= self.capacity,
            "Dynamic delay discontinuity roots exceed max_discontinuities.",
        )
        root_times = state.root_times.at[state.num_roots].set(checked_time)
        state = eqx.tree_at(
            lambda item: (
                item.processed,
                item.bracket_active,
                item.root_times,
                item.num_roots,
                item.num_restarts,
            ),
            state,
            (
                processed,
                jnp.asarray(False),
                root_times,
                state.num_roots + 1,
                state.num_restarts + 1,
            ),
        )
        state, _ = self._insert_source(
            state,
            checked_time,
            generation,
            jnp.asarray(True),
        )
        return self._append_constant_children(state, checked_time, generation)


def _dynamic_adapt(
    tracker: StateDependentDiscontinuityTracker,
    t0: Array,
    t1: Array,
    y0: Array,
    y1: Array,
    args: Any,
    base_output: tuple[Any, ...],
    old_state: DynamicControllerState,
) -> tuple[Any, ...]:
    base_keep, base_t0, base_t1, base_jump, new_inner, base_result = base_output
    discontinuities = old_state.discontinuities

    def bracket_branch(_):
        def rejected_by_error(_):
            next_time = tracker.cap_next(discontinuities, t0, base_t1)
            return (
                base_keep,
                base_t0,
                next_time,
                base_jump,
                DynamicControllerState(new_inner, discontinuities),
                base_result,
            )

        def refine(_):
            refined, next_time, accept_root, below_root = tracker.refine_bracket(
                discontinuities,
                t1,
                y1,
                args,
            )

            def accept(_):
                accepted = tracker.accept_root(refined, t1, y1, args)
                next_end = tracker.cap_next(accepted, base_t0, base_t1)
                return (
                    jnp.asarray(True),
                    base_t0,
                    next_end,
                    jnp.asarray(True),
                    DynamicControllerState(new_inner, accepted),
                    base_result,
                )

            def accept_before_root(_):
                next_end = tracker.cap_next(refined, base_t0, next_time)
                return (
                    jnp.asarray(True),
                    base_t0,
                    next_end,
                    base_jump,
                    DynamicControllerState(new_inner, refined),
                    base_result,
                )

            def reject(_):
                return (
                    jnp.asarray(False),
                    t0,
                    jax.lax.stop_gradient(next_time),
                    jnp.asarray(False),
                    DynamicControllerState(old_state.inner_state, refined),
                    dfx.RESULTS.successful,
                )

            return jax.lax.cond(
                accept_root,
                accept,
                lambda _: jax.lax.cond(
                    below_root,
                    accept_before_root,
                    reject,
                    operand=None,
                ),
                operand=None,
            )

        return jax.lax.cond(base_keep, refine, rejected_by_error, operand=None)

    def ordinary_branch(_):
        bracketed, next_time, has_crossing = tracker.start_bracket(
            discontinuities,
            t0,
            t1,
            y0,
            y1,
            args,
        )

        def start(_):
            return (
                jnp.asarray(False),
                t0,
                jax.lax.stop_gradient(next_time),
                jnp.asarray(False),
                DynamicControllerState(old_state.inner_state, bracketed),
                dfx.RESULTS.successful,
            )

        def no_root(_):
            landed, landed_any = jax.lax.cond(
                base_keep,
                lambda current: tracker.mark_pending_landed(current, t1),
                lambda current: (current, jnp.asarray(False)),
                discontinuities,
            )
            use_landed = base_keep & landed_any
            selected = landed
            next_end = tracker.cap_next(selected, base_t0, base_t1)
            return (
                base_keep,
                base_t0,
                next_end,
                base_jump | use_landed,
                DynamicControllerState(new_inner, selected),
                base_result,
            )

        return jax.lax.cond(base_keep & has_crossing, start, no_root, operand=None)

    return jax.lax.cond(
        discontinuities.bracket_active,
        bracket_branch,
        ordinary_branch,
        operand=None,
    )


class StateDependentAdaptiveController(dfx.AbstractAdaptiveStepSizeController):
    """Adaptive controller adding certified internal root/restart handling."""

    controller: dfx.AbstractAdaptiveStepSizeController
    tracker: StateDependentDiscontinuityTracker

    def __init__(
        self,
        controller: dfx.AbstractAdaptiveStepSizeController,
        tracker: StateDependentDiscontinuityTracker,
    ):
        self.controller = controller
        self.tracker = tracker

    @property
    def rtol(self):
        return self.controller.rtol

    @property
    def atol(self):
        return self.controller.atol

    @property
    def norm(self):
        return self.controller.norm

    def wrap(self, direction):
        return StateDependentAdaptiveController(
            cast(
                dfx.AbstractAdaptiveStepSizeController,
                self.controller.wrap(direction),
            ),
            self.tracker,
        )

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        next_time, inner_state = self.controller.init(
            terms,
            t0,
            t1,
            y0,
            dt0,
            args,
            func,
            error_order,
        )
        discontinuities = self.tracker.initial_state(t0, y0, args)
        next_time = self.tracker.cap_next(discontinuities, t0, next_time)
        return next_time, DynamicControllerState(inner_state, discontinuities)

    def adapt_step_size(
        self,
        t0,
        t1,
        y0,
        y1_candidate,
        args,
        y_error,
        error_order,
        controller_state,
    ):
        base_output = self.controller.adapt_step_size(
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            y_error,
            error_order,
            controller_state.inner_state,
        )
        return _dynamic_adapt(
            self.tracker,
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            base_output,
            controller_state,
        )


class StateDependentFixedController(dfx.AbstractStepSizeController):
    """Fixed controller with low-allocation dynamic root/restart handling."""

    controller: dfx.AbstractStepSizeController
    tracker: StateDependentDiscontinuityTracker

    def __init__(
        self,
        controller: dfx.AbstractStepSizeController,
        tracker: StateDependentDiscontinuityTracker,
    ):
        self.controller = controller
        self.tracker = tracker

    def wrap(self, direction):
        return StateDependentFixedController(
            self.controller.wrap(direction), self.tracker
        )

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        next_time, inner_state = self.controller.init(
            terms,
            t0,
            t1,
            y0,
            dt0,
            args,
            func,
            error_order,
        )
        discontinuities = self.tracker.initial_state(t0, y0, args)
        next_time = self.tracker.cap_next(discontinuities, t0, next_time)
        return next_time, DynamicControllerState(inner_state, discontinuities)

    def adapt_step_size(
        self,
        t0,
        t1,
        y0,
        y1_candidate,
        args,
        y_error,
        error_order,
        controller_state,
    ):
        base_output = self.controller.adapt_step_size(
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            y_error,
            error_order,
            controller_state.inner_state,
        )
        return _dynamic_adapt(
            self.tracker,
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            base_output,
            controller_state,
        )


__all__ = [
    "DynamicControllerState",
    "StateDependentAdaptiveController",
    "StateDependentDiscontinuityTracker",
    "StateDependentFixedController",
    "constant_discontinuity_schedule",
]
