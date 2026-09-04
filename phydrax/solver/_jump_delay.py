#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..stochastic import JUMP_SUCCESS, JumpEventBatch, WienerRealization
from ._delay import DelayDifferentialProblem, DelayHistory, DelayValues, DerivativeDelay
from ._delay_history import DelayHistoryView
from ._delay_segmented import DelaySegmentArchive
from ._diffrax_delay_backend import _DelayVectorField, solve_diffrax_delay
from ._diffrax_state_packing import (
    _prepare_diffrax_state_adapter,
    DiffraxComplexStatePolicy,
)
from ._memory import MemoryEquationSolution
from ._save_schedule import validate_save_times


DelayJumpMap: TypeAlias = Callable[
    [Array, Array, DelayValues, Array, Array, Any], ArrayLike
]


class JumpDelayProblem(StrictModule):
    """Retarded delay equation with prescribed finite-activity state jumps.

    The jump map receives ``(time, pre_state, memory, channel, mark, args)``. Continuous
    dynamics and optional Wiener terms are inherited from ``delay_problem``. Event
    times are supplied separately as a ``JumpEventBatch`` and are interpreted as an
    exogenous, right-continuous schedule.
    """

    delay_problem: DelayDifferentialProblem
    jump: DelayJumpMap
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        delay_problem: DelayDifferentialProblem,
        jump: DelayJumpMap,
        /,
        *,
        mark_shape: Sequence[int] = (),
        problem_id: str = "jump-delay-problem",
    ):
        if not isinstance(delay_problem, DelayDifferentialProblem):
            raise TypeError("delay_problem must be a DelayDifferentialProblem.")
        if not callable(jump):
            raise TypeError("jump must be callable.")
        if any(isinstance(term, DerivativeDelay) for term in delay_problem.delay_terms):
            raise ValueError(
                "Jump-delay execution does not support derivative-valued delay terms."
            )
        shape = tuple(int(size) for size in mark_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("mark_shape dimensions must be positive.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.delay_problem = delay_problem
        self.jump = jump
        self.mark_shape = shape
        self.problem_id = identifier

    @property
    def t0(self) -> Array:
        return self.delay_problem.t0

    @property
    def t1(self) -> Array:
        return self.delay_problem.t1

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.delay_problem.state_shape

    @property
    def stochastic(self) -> bool:
        return self.delay_problem.stochastic

    @property
    def noise_shape(self) -> tuple[int, ...]:
        return self.delay_problem.noise_shape

    @property
    def noise_id(self) -> str | None:
        return self.delay_problem.noise_id


class JumpDelayBackendResult(StrictModule):
    """Continuous interval results plus resolved pre/post jump states."""

    continuous_results: tuple[Any, ...]
    events: JumpEventBatch


class _RestartHistory(eqx.Module):
    history: DelayHistoryView
    restart_time: Array
    restart_state: Array

    @eqx.filter_jit
    def __call__(self, time: Array, args: Any) -> Array:
        del args
        tolerance = (
            100.0
            * jnp.finfo(jnp.result_type(time, float)).eps
            * jnp.maximum(1.0, jnp.abs(self.restart_time))
        )
        return jax.lax.cond(
            time >= self.restart_time - tolerance,
            lambda _: self.restart_state,
            lambda query: self.history.value(
                jnp.minimum(query, self.restart_time), left=False
            ),
            time,
        )


def _archive(
    starts: list[float], ends: list[float], interpolations: list[Any]
) -> DelaySegmentArchive:
    return DelaySegmentArchive(
        starts=np.asarray(starts, dtype=float),
        ends=np.asarray(ends, dtype=float),
        interpolations=tuple(interpolations),
    )


def _restart_history(
    problem: DelayDifferentialProblem,
    archive: DelaySegmentArchive,
    time: Array,
    state: Array,
) -> _RestartHistory:
    return _RestartHistory(
        history=DelayHistoryView(
            initial_history=problem.history,
            initial_derivative=problem.history_derivative,
            args=problem.args,
            initial_time=problem.t0,
            computed_history=archive,
            state_shape=problem.state_shape,
            geometry=problem.state_geometry,
            derivative_shape=problem.tangent_shape,
        ),
        restart_time=time,
        restart_state=state,
    )


def _interval_problem(
    problem: DelayDifferentialProblem,
    history: DelayHistory,
    start: Array,
    end: Array,
) -> DelayDifferentialProblem:
    return DelayDifferentialProblem(
        problem.drift,
        history,
        problem.delay_terms,
        t0=start,
        t1=end,
        args=problem.args,
        wiener_terms=problem.wiener_terms,
        interpretation=problem.interpretation,
        state_geometry=problem.state_geometry,
        problem_id=problem.problem_id,
    )


@eqx.filter_jit
def _event_memory(
    problem: DelayDifferentialProblem,
    archive: DelaySegmentArchive,
    time: Array,
    state: Array,
) -> DelayValues:
    state_adapter = _prepare_diffrax_state_adapter(
        problem.initial_state,
        DiffraxComplexStatePolicy("native"),
        None,
        problem.state_geometry,
    )
    context = _DelayVectorField(
        function=problem.drift,
        initial_history=problem.history,
        initial_derivative=problem.history_derivative,
        delay_terms=problem.delay_terms,
        initial_time=problem.t0,
        state_shape=problem.state_shape,
        tangent_shape=problem.tangent_shape,
        geometry=problem.state_geometry,
        state_adapter=state_adapter,
        backend_shape=state_adapter.backend_shape,
        backend_tangent_shape=problem.tangent_shape,
        computed_history=archive,
    )
    memory, _ = context._memory(time, state, problem.args)
    return memory


def _validated_schedule(
    problem: JumpDelayProblem, events: JumpEventBatch
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(events, JumpEventBatch):
        raise TypeError("events must be a JumpEventBatch.")
    if events.batch_shape:
        raise ValueError("Jump-delay execution currently requires one event stream.")
    if events.mark_shape != problem.mark_shape:
        raise ValueError("Jump event mark_shape does not match the problem.")
    status = int(np.asarray(jax.device_get(events.status)))
    if status != JUMP_SUCCESS:
        raise ValueError("Jump-delay execution requires a successful event schedule.")
    valid = np.asarray(jax.device_get(events.valid), dtype=bool)
    count = int(np.sum(valid))
    if np.any(valid[count:]):
        raise ValueError("Jump event validity must form one leading prefix.")
    times = np.asarray(jax.device_get(events.times[:count]), dtype=float)
    channels = np.asarray(jax.device_get(events.channels[:count]), dtype=int)
    base = problem.delay_problem
    if (
        np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or np.any(times <= float(base.t0))
        or np.any(times >= float(base.t1))
    ):
        raise ValueError(
            "Valid jump times must be finite, strictly increasing, and lie inside "
            "the open solve interval."
        )
    if np.any(channels < 0):
        raise ValueError("Valid jump channels must be nonnegative.")
    return times, channels


def solve_jump_delay(
    problem: JumpDelayProblem,
    events: JumpEventBatch,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization | None = None,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense: bool = False,
    max_steps: int = 4096,
    initial_discontinuities: ArrayLike | Sequence[float] | None = None,
    discontinuity_depth: int | None = None,
    max_discontinuities: int = 8192,
    root_rtol: float = 1e-10,
    root_atol: float = 1e-12,
    max_root_iterations: int = 64,
) -> MemoryEquationSolution:
    """Integrate exact continuous delay intervals and apply prescribed jumps."""
    if not isinstance(problem, JumpDelayProblem):
        raise TypeError("problem must be a JumpDelayProblem.")
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    if realization is not None and realization.sample_shape:
        raise ValueError("Jump-delay Wiener realizations must contain one path.")
    base = problem.delay_problem
    requested_times = validate_save_times(base.t0, base.t1, save_times)
    event_times, event_channels = _validated_schedule(problem, events)
    event_count = int(event_times.size)
    pre_states = jnp.zeros(
        (events.max_events,) + base.state_shape, dtype=base.initial_state.dtype
    )
    post_states = jnp.zeros_like(pre_states)
    starts: list[float] = []
    ends: list[float] = []
    interpolations: list[Any] = []
    continuous_results: list[Any] = []
    total_steps: Any = 0
    total_accepted: Any = 0
    total_rejected: Any = 0
    current_time = base.t0
    current_history: DelayHistory = base.history
    declared_sources = (
        jnp.empty((0,), dtype=base.t0.dtype)
        if initial_discontinuities is None
        else jnp.asarray(initial_discontinuities, dtype=base.t0.dtype)
    )

    boundaries = tuple(event_times.tolist()) + (float(base.t1),)
    for interval_index, boundary in enumerate(boundaries):
        end = jnp.asarray(boundary, dtype=base.t1.dtype)
        interval = _interval_problem(base, current_history, current_time, end)
        propagated_sources = jnp.concatenate(
            (
                declared_sources,
                jnp.asarray(event_times[:interval_index], dtype=base.t0.dtype),
            )
        )
        continuous = solve_diffrax_delay(
            interval,
            save_times=jnp.stack((current_time, end)),
            realization=realization,
            solver=solver,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            dt0=dt0,
            rtol=rtol,
            atol=atol,
            dense=True,
            history_mode="full",
            max_steps=max_steps,
            initial_discontinuities=propagated_sources,
            discontinuity_depth=discontinuity_depth,
            max_discontinuities=max_discontinuities,
            root_rtol=root_rtol,
            root_atol=root_atol,
            max_root_iterations=max_root_iterations,
            throw=True,
        )
        assert continuous.interpolation is not None
        starts.append(float(current_time))
        ends.append(boundary)
        interpolations.append(continuous.interpolation)
        continuous_results.append(continuous.backend_result)
        total_steps = total_steps + continuous.stats["num_steps"]
        total_accepted = total_accepted + continuous.stats["num_accepted_steps"]
        total_rejected = total_rejected + continuous.stats["num_rejected_steps"]
        archive = _archive(starts, ends, interpolations)

        if interval_index < event_count:
            event_time = end
            pre_state = continuous.states[-1]
            memory = _event_memory(base, archive, event_time, pre_state)
            channel = events.channels[interval_index]
            mark = events.marks[interval_index]
            post_state = jnp.asarray(
                problem.jump(
                    event_time,
                    pre_state,
                    memory,
                    channel,
                    mark,
                    base.args,
                )
            )
            if post_state.shape != base.state_shape:
                raise ValueError("Jump map must preserve the delay state shape.")
            post_state = eqx.error_if(
                post_state,
                ~jnp.all(jnp.isfinite(post_state)),
                "Jump map returned a nonfinite state.",
            )
            if base.state_geometry is not None:
                membership = jnp.asarray(
                    base.state_geometry.contains(post_state), dtype=bool
                )
                if membership.shape != ():
                    raise ValueError(
                        "State geometry contains() must return a scalar boolean."
                    )
                post_state = eqx.error_if(
                    post_state,
                    ~membership,
                    "Jump map returned a state outside state_geometry.",
                )
            pre_states = pre_states.at[interval_index].set(pre_state)
            post_states = post_states.at[interval_index].set(post_state)
            current_time = event_time
            current_history = _restart_history(base, archive, event_time, post_state)

    final_archive = _archive(starts, ends, interpolations)
    states = final_archive.evaluate(requested_times, left=False)
    valid = jnp.all(jnp.isfinite(states), axis=tuple(range(1, states.ndim)))
    resolved_events = JumpEventBatch(
        events.times,
        events.channels,
        events.marks,
        events.valid,
        events.status,
        mark_shape=events.mark_shape,
        state_shape=base.state_shape,
        pre_states=pre_states,
        post_states=post_states,
    )
    selected_name = continuous.solver_name
    driver_family = (
        "wiener-plus-finite-activity-jump" if base.stochastic else "finite-activity-jump"
    )
    return MemoryEquationSolution(
        times=requested_times,
        states=states,
        valid=valid,
        interpolation=final_archive if dense else None,
        backend_result=JumpDelayBackendResult(
            continuous_results=tuple(continuous_results),
            events=resolved_events,
        ),
        stats={
            "num_steps": total_steps,
            "num_accepted_steps": total_accepted,
            "num_rejected_steps": total_rejected,
            "num_jumps": event_count,
            "num_continuous_intervals": event_count + 1,
        },
        event_mask=events.valid,
        realization=realization,
        state_shape=base.state_shape,
        solver_name=f"{selected_name}JumpDelay",
        solver_id=f"solver:diffrax-jump-delay:{selected_name}:host-hybrid-v1",
        resolved_method=f"{selected_name}:exact-jump-time-method-of-steps",
        metadata={
            "problem_id": problem.problem_id,
            "delay_problem_id": base.problem_id,
            "backend": "diffrax-host-hybrid",
            "equation_kind": "jump-delay",
            "driver_family": driver_family,
            "jump_side_convention": "right-continuous",
            "jump_times": events.times,
            "jump_channels": events.channels,
            "state_geometry_id": base.state_geometry_id,
        },
    )


__all__ = [
    "DelayJumpMap",
    "JumpDelayBackendResult",
    "JumpDelayProblem",
    "solve_jump_delay",
]
