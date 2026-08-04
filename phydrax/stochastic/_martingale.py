#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ._jump import AbstractJumpProcess
from ._trajectory import StochasticTrajectory


def jump_generator_observable(
    process: AbstractJumpProcess,
    state: ArrayLike,
    /,
    *,
    time: ArrayLike,
    observable: Callable[[Array], Array],
    key: Key[Array, ""],
    num_mark_samples: int = 1,
    args: Any = None,
) -> Array:
    """Evaluate a finite-activity jump generator on an observable.

    Random marks are integrated by explicit Monte Carlo. Deterministic marks therefore
    need only one sample, while random-mark calculations retain their key and sample
    count at the call site instead of pretending the approximation is exact.
    """
    if not isinstance(process, AbstractJumpProcess):
        raise TypeError("process must implement AbstractJumpProcess.")
    if not callable(observable):
        raise TypeError("observable must be callable.")
    count = int(num_mark_samples)
    if count <= 0:
        raise ValueError("num_mark_samples must be positive.")
    state_array = jnp.asarray(state)
    if state_array.shape != process.state_shape:
        raise ValueError(
            f"state must have process state shape {process.state_shape}; "
            f"got {state_array.shape}."
        )
    t = jnp.asarray(time)
    if t.shape != ():
        raise ValueError("time must be scalar.")
    rates = jnp.asarray(process.intensities(t, state_array, args))
    if rates.shape != (process.num_channels,):
        raise ValueError(
            "Jump intensities must have shape "
            f"{(process.num_channels,)}; got {rates.shape}."
        )
    rates = eqx.error_if(
        rates,
        jnp.any(~jnp.isfinite(rates)) | jnp.any(rates < 0.0),
        "Jump intensities must be finite and nonnegative.",
    )
    base = jnp.asarray(observable(state_array))
    channels = jnp.arange(process.num_channels, dtype=jnp.int32)
    channel_keys = jr.split(key, process.num_channels)

    def one_channel(channel: Array, channel_key: Array) -> Array:
        mark_keys = jr.split(channel_key, count)

        def one_mark(mark_key: Array) -> Array:
            mark = process.sample_mark(mark_key, t, state_array, channel, args)
            post_state = process.jump(state_array, channel, mark, args)
            return jnp.asarray(observable(post_state)) - base

        return jnp.mean(jax.vmap(one_mark)(mark_keys), axis=0)

    increments = jax.vmap(one_channel)(channels, channel_keys)
    return jnp.tensordot(rates, increments, axes=((0,), (0,)))


MartingaleQuadrature: TypeAlias = Literal["left", "midpoint", "trapezoid"]
MartingaleReduction: TypeAlias = Literal["mean", "sum", "none"]


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError(f"{owner} dimensions must be positive.")
    return resolved


class MartingaleProblem(StrictModule):
    """Observable and declared generator action defining one martingale problem."""

    observable: Callable[[Array], Array]
    generator_observable: Callable[[Array, Array], Array]
    bracket_density: Callable[[Array, Array], Array] | None
    observable_shape: tuple[int, ...] = eqx.field(static=True)
    label: str = eqx.field(static=True)

    def __init__(
        self,
        observable: Callable[[Array], Array],
        generator_observable: Callable[[Array, Array], Array],
        /,
        *,
        observable_shape: Sequence[int] = (),
        bracket_density: Callable[[Array, Array], Array] | None = None,
        label: str = "martingale",
    ):
        if not callable(observable) or not callable(generator_observable):
            raise TypeError("observable and generator_observable must be callable.")
        if bracket_density is not None and not callable(bracket_density):
            raise TypeError("bracket_density must be callable or None.")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string.")
        self.observable = observable
        self.generator_observable = generator_observable
        self.bracket_density = bracket_density
        self.observable_shape = _shape(observable_shape, owner="observable_shape")
        self.label = label


class MartingaleIncrements(StrictModule):
    """Interval and cumulative martingale residuals over a stochastic trajectory."""

    observable_values: Array
    generator_values: Array
    compensator_increments: Array
    increments: Array
    cumulative: Array
    interval_valid: Array
    trajectory: StochasticTrajectory
    problem: MartingaleProblem
    observable_shape: tuple[int, ...] = eqx.field(static=True)
    quadrature: MartingaleQuadrature = eqx.field(static=True)

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return self.trajectory.leading_shape

    @property
    def num_intervals(self) -> int:
        return self.trajectory.num_times - 1


class StoppingIndices(StrictModule):
    """First bounded stopping node for every trajectory, with explicit hit status."""

    indices: Array
    hit: Array
    label: str = eqx.field(static=True)


def _evaluate_state_time(
    function: Callable[[Array, Array], Array],
    states: Array,
    times: Array,
    valid: Array,
    /,
    *,
    state_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    owner: str,
) -> Array:
    flat_states = states.reshape((-1,) + state_shape)
    flat_times = times.reshape((-1,))
    flat_valid = valid.reshape((-1,))
    safe_states = jnp.where(
        flat_valid.reshape((-1,) + (1,) * len(state_shape)),
        flat_states,
        jnp.zeros_like(flat_states),
    )
    values = jax.vmap(lambda state, time: jnp.asarray(function(state, time)))(
        safe_states, flat_times
    )
    if values.shape[1:] != output_shape:
        raise ValueError(
            f"{owner} must return observable shape {output_shape}; "
            f"got {values.shape[1:]}."
        )
    return values.reshape(times.shape + output_shape)


def martingale_increments(
    trajectory: StochasticTrajectory,
    problem: MartingaleProblem,
    /,
    *,
    quadrature: MartingaleQuadrature = "left",
    compensator_increments: ArrayLike | None = None,
) -> MartingaleIncrements:
    """Evaluate discrete martingale residuals without erasing trajectory provenance."""
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    if not isinstance(problem, MartingaleProblem):
        raise TypeError("problem must be a MartingaleProblem.")
    if trajectory.num_times < 2:
        raise ValueError("Martingale increments require at least two saved times.")
    if quadrature not in ("left", "midpoint", "trapezoid"):
        raise ValueError("quadrature must be 'left', 'midpoint', or 'trapezoid'.")
    solution_spec = trajectory.metadata.get("spde_solution_spec")
    if solution_spec is not None:
        from ._solution import SPDESolutionSpec

        if not isinstance(solution_spec, SPDESolutionSpec):
            raise TypeError("spde_solution_spec metadata must be an SPDESolutionSpec.")
        solution_spec.assert_supports("martingale")

    state_shape = trajectory.state_shape
    output_shape = problem.observable_shape
    observable_values = _evaluate_state_time(
        lambda state, _time: problem.observable(state),
        trajectory.states,
        trajectory.times,
        trajectory.valid,
        state_shape=state_shape,
        output_shape=output_shape,
        owner="observable",
    )
    generator_values = _evaluate_state_time(
        problem.generator_observable,
        trajectory.states,
        trajectory.times,
        trajectory.valid,
        state_shape=state_shape,
        output_shape=output_shape,
        owner="generator_observable",
    )
    time_axis = len(trajectory.leading_shape)
    dt = jnp.diff(trajectory.times, axis=-1)
    dt_expanded = dt.reshape(dt.shape + (1,) * len(output_shape))
    if compensator_increments is not None:
        expected = trajectory.leading_shape + (trajectory.num_times - 1,) + output_shape
        compensator = jnp.asarray(compensator_increments)
        if compensator.shape != expected:
            raise ValueError(
                f"compensator_increments must have shape {expected}; "
                f"got {compensator.shape}."
            )
    elif quadrature == "left":
        compensator = (
            jnp.take(
                generator_values,
                jnp.arange(trajectory.num_times - 1),
                axis=time_axis,
            )
            * dt_expanded
        )
    elif quadrature == "trapezoid":
        left = jnp.take(
            generator_values,
            jnp.arange(trajectory.num_times - 1),
            axis=time_axis,
        )
        right = jnp.take(
            generator_values,
            jnp.arange(1, trajectory.num_times),
            axis=time_axis,
        )
        compensator = 0.5 * (left + right) * dt_expanded
    else:
        left_states = jnp.take(
            trajectory.states,
            jnp.arange(trajectory.num_times - 1),
            axis=time_axis,
        )
        right_states = jnp.take(
            trajectory.states,
            jnp.arange(1, trajectory.num_times),
            axis=time_axis,
        )
        midpoint_states = 0.5 * (left_states + right_states)
        midpoint_times = 0.5 * (trajectory.times[..., :-1] + trajectory.times[..., 1:])
        midpoint_valid = trajectory.valid[..., :-1] & trajectory.valid[..., 1:]
        midpoint_generator = _evaluate_state_time(
            problem.generator_observable,
            midpoint_states,
            midpoint_times,
            midpoint_valid,
            state_shape=state_shape,
            output_shape=output_shape,
            owner="generator_observable",
        )
        compensator = midpoint_generator * dt_expanded

    left_values = jnp.take(
        observable_values,
        jnp.arange(trajectory.num_times - 1),
        axis=time_axis,
    )
    right_values = jnp.take(
        observable_values,
        jnp.arange(1, trajectory.num_times),
        axis=time_axis,
    )
    increments = right_values - left_values - compensator
    interval_valid = (
        trajectory.valid[..., :-1]
        & trajectory.valid[..., 1:]
        & jnp.all(
            jnp.isfinite(increments),
            axis=tuple(range(len(trajectory.leading_shape) + 1, increments.ndim)),
        )
    )
    safe_increments = jnp.where(
        interval_valid.reshape(interval_valid.shape + (1,) * len(output_shape)),
        increments,
        jnp.zeros_like(increments),
    )
    zero = jnp.zeros(
        trajectory.leading_shape + (1,) + output_shape,
        dtype=increments.dtype,
    )
    cumulative = jnp.concatenate(
        (zero, jnp.cumsum(safe_increments, axis=time_axis)),
        axis=time_axis,
    )
    return MartingaleIncrements(
        observable_values=observable_values,
        generator_values=generator_values,
        compensator_increments=compensator,
        increments=increments,
        cumulative=cumulative,
        interval_valid=interval_valid,
        trajectory=trajectory,
        problem=problem,
        observable_shape=output_shape,
        quadrature=quadrature,
    )


def first_stopping_indices(
    trajectory: StochasticTrajectory,
    condition: Callable[[Array, Array], ArrayLike],
    /,
    *,
    label: str = "stopping-time",
) -> StoppingIndices:
    """Find the first saved node satisfying a user-declared stopping condition."""
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    if not callable(condition):
        raise TypeError("condition must be callable.")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string.")
    state_shape = trajectory.state_shape
    flat_states = trajectory.states.reshape((-1,) + state_shape)
    flat_times = trajectory.times.reshape((-1,))
    values = jax.vmap(
        lambda state, time: jnp.asarray(condition(state, time), dtype=bool)
    )(flat_states, flat_times)
    if values.shape != flat_times.shape:
        raise ValueError("A stopping condition must return one scalar boolean.")
    values = values.reshape(trajectory.leading_shape + (trajectory.num_times,))
    eligible = values & trajectory.valid
    hit = jnp.any(eligible, axis=-1)
    indices = jnp.where(
        hit,
        jnp.argmax(eligible, axis=-1),
        trajectory.num_times - 1,
    ).astype(jnp.int32)
    return StoppingIndices(indices=indices, hit=hit, label=label)


def stopped_martingale_increments(
    residuals: MartingaleIncrements,
    stopping: StoppingIndices,
    /,
) -> MartingaleIncrements:
    """Stop each martingale at a bounded saved node and hold it constant afterward."""
    if stopping.indices.shape != residuals.leading_shape:
        raise ValueError("Stopping indices must have the trajectory leading shape.")
    active = jnp.arange(residuals.num_intervals) < stopping.indices[..., None]
    valid = residuals.interval_valid & active
    mask = valid.reshape(valid.shape + (1,) * len(residuals.observable_shape))
    increments = jnp.where(mask, residuals.increments, 0.0)
    compensator = jnp.where(mask, residuals.compensator_increments, 0.0)
    time_axis = len(residuals.leading_shape)
    zero = jnp.zeros(
        residuals.leading_shape + (1,) + residuals.observable_shape,
        dtype=increments.dtype,
    )
    cumulative = jnp.concatenate(
        (zero, jnp.cumsum(increments, axis=time_axis)),
        axis=time_axis,
    )
    return MartingaleIncrements(
        observable_values=residuals.observable_values,
        generator_values=residuals.generator_values,
        compensator_increments=compensator,
        increments=increments,
        cumulative=cumulative,
        interval_valid=valid,
        trajectory=residuals.trajectory,
        problem=residuals.problem,
        observable_shape=residuals.observable_shape,
        quadrature=residuals.quadrature,
    )


def quadratic_covariation(
    residuals: MartingaleIncrements,
    /,
    *,
    cumulative: bool = True,
) -> Array:
    """Compute optional quadratic covariation over flattened observable components."""
    event_size = prod(residuals.observable_shape) if residuals.observable_shape else 1
    flat = residuals.increments.reshape(
        residuals.leading_shape + (residuals.num_intervals, event_size)
    )
    values = flat[..., :, :, None] * flat[..., :, None, :]
    values = jnp.where(
        residuals.interval_valid[..., None, None],
        values,
        0.0,
    )
    if not cumulative:
        return values
    axis = len(residuals.leading_shape)
    zero = jnp.zeros(
        residuals.leading_shape + (1, event_size, event_size),
        dtype=values.dtype,
    )
    return jnp.concatenate((zero, jnp.cumsum(values, axis=axis)), axis=axis)


def predictable_bracket_increments(
    residuals: MartingaleIncrements,
    /,
) -> Array:
    """Integrate a declared predictable bracket density on residual intervals."""
    density = residuals.problem.bracket_density
    if density is None:
        raise ValueError("This martingale problem does not declare bracket_density.")
    trajectory = residuals.trajectory
    event_size = prod(residuals.observable_shape) if residuals.observable_shape else 1
    matrix_shape = (event_size, event_size)

    def evaluate(states: Array, times: Array, valid: Array) -> Array:
        flat_states = states.reshape((-1,) + trajectory.state_shape)
        flat_times = times.reshape((-1,))
        flat_valid = valid.reshape((-1,))
        safe_states = jnp.where(
            flat_valid.reshape((-1,) + (1,) * len(trajectory.state_shape)),
            flat_states,
            jnp.zeros_like(flat_states),
        )
        values = jax.vmap(
            lambda state, time: jnp.asarray(density(state, time)).reshape(matrix_shape)
        )(safe_states, flat_times)
        return values.reshape(times.shape + matrix_shape)

    time_axis = len(trajectory.leading_shape)
    dt = jnp.diff(trajectory.times, axis=-1)[..., None, None]
    if residuals.quadrature == "left":
        values = evaluate(
            jnp.take(
                trajectory.states,
                jnp.arange(residuals.num_intervals),
                axis=time_axis,
            ),
            trajectory.times[..., :-1],
            residuals.interval_valid,
        )
    elif residuals.quadrature == "trapezoid":
        left = evaluate(
            jnp.take(
                trajectory.states,
                jnp.arange(residuals.num_intervals),
                axis=time_axis,
            ),
            trajectory.times[..., :-1],
            residuals.interval_valid,
        )
        right = evaluate(
            jnp.take(
                trajectory.states,
                jnp.arange(1, trajectory.num_times),
                axis=time_axis,
            ),
            trajectory.times[..., 1:],
            residuals.interval_valid,
        )
        values = 0.5 * (left + right)
    else:
        left_states = jnp.take(
            trajectory.states,
            jnp.arange(residuals.num_intervals),
            axis=time_axis,
        )
        right_states = jnp.take(
            trajectory.states,
            jnp.arange(1, trajectory.num_times),
            axis=time_axis,
        )
        values = evaluate(
            0.5 * (left_states + right_states),
            0.5 * (trajectory.times[..., :-1] + trajectory.times[..., 1:]),
            residuals.interval_valid,
        )
    return jnp.where(
        residuals.interval_valid[..., None, None],
        values * dt,
        0.0,
    )


def carre_du_champ(
    generator_action: Callable[[Callable[[Array], Array], Array, Array], Array],
    observable: Callable[[Array], Array],
    state: ArrayLike,
    time: ArrayLike,
    /,
) -> Array:
    """Evaluate Γ(φ)=L(φ²)−2φLφ from a reusable generator action."""
    if not callable(generator_action) or not callable(observable):
        raise TypeError("generator_action and observable must be callable.")
    state_array = jnp.asarray(state)
    time_array = jnp.asarray(time)
    value = jnp.asarray(observable(state_array))
    first = jnp.asarray(generator_action(observable, state_array, time_array))
    second = jnp.asarray(
        generator_action(
            lambda item: jnp.asarray(observable(item)) ** 2, state_array, time_array
        )
    )
    return second - 2.0 * value * first


def combined_generator_observable(
    *terms: Callable[[Array, Array], Array],
) -> Callable[[Array, Array], Array]:
    """Compose additive generator contributions without hiding their ownership."""
    if not terms or any(not callable(term) for term in terms):
        raise ValueError("combined_generator_observable requires callable terms.")

    def evaluate(state: Array, time: Array) -> Array:
        values = tuple(jnp.asarray(term(state, time)) for term in terms)
        return sum(values[1:], start=values[0])

    return evaluate


def martingale_moment_loss(
    residuals: MartingaleIncrements,
    instruments: Sequence[Callable[[Array, Array], ArrayLike]] = (),
    /,
    *,
    reduction: MartingaleReduction = "mean",
) -> Array:
    """Penalize predictable-instrument martingale moments differentiably."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    resolved = (
        (lambda _state, _time: jnp.asarray(1.0),)
        if not instruments
        else tuple(instruments)
    )
    if any(not callable(instrument) for instrument in resolved):
        raise TypeError("instruments must contain callables.")
    trajectory = residuals.trajectory
    time_axis = len(trajectory.leading_shape)
    source_states = jnp.take(
        trajectory.states,
        jnp.arange(residuals.num_intervals),
        axis=time_axis,
    )
    source_times = trajectory.times[..., :-1]
    flat_states = source_states.reshape((-1,) + trajectory.state_shape)
    flat_times = source_times.reshape((-1,))
    event_shape = residuals.observable_shape
    moment_terms = []
    for instrument in resolved:
        values = jax.vmap(lambda state, time: jnp.asarray(instrument(state, time)))(
            flat_states, flat_times
        )
        if values.shape[1:] not in ((), event_shape):
            raise ValueError(
                "Predictable instruments must return scalars or observable-shaped arrays."
            )
        values = values.reshape(source_times.shape + values.shape[1:])
        if values.shape[1 + len(trajectory.leading_shape) :] == ():
            values = values.reshape(values.shape + (1,) * len(event_shape))
        products = values * residuals.increments
        mask = residuals.interval_valid.reshape(
            residuals.interval_valid.shape + (1,) * len(event_shape)
        )
        count = jnp.sum(mask)
        moment = jnp.sum(
            jnp.where(mask, products, 0.0),
            axis=tuple(range(len(trajectory.leading_shape) + 1)),
        )
        moment_terms.append(moment / jnp.maximum(count, 1))
    losses = jnp.abs(jnp.stack(moment_terms, axis=0)) ** 2
    if reduction == "none":
        return losses
    if reduction == "sum":
        return jnp.sum(losses)
    return jnp.mean(losses)


__all__ = [
    "carre_du_champ",
    "combined_generator_observable",
    "first_stopping_indices",
    "jump_generator_observable",
    "martingale_increments",
    "martingale_moment_loss",
    "MartingaleIncrements",
    "MartingaleProblem",
    "MartingaleQuadrature",
    "MartingaleReduction",
    "predictable_bracket_increments",
    "quadratic_covariation",
    "stopped_martingale_increments",
    "StoppingIndices",
]
