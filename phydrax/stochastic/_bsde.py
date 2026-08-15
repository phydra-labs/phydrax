#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._frozendict import frozendict
from .._strict import StrictModule
from ._jump import JumpEventBatch
from ._realization import is_stochastic_realization, StochasticRealization
from ._wiener import WienerRealization


BSDEQuadrature: TypeAlias = Literal["left", "trapezoid"]
BSDEObjectiveMode: TypeAlias = Literal["terminal", "local", "global", "joint"]
BSDEControlMode: TypeAlias = Literal["explicit", "autodiff"]


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _event_finite(values: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return jnp.isfinite(values)
    axes = tuple(range(values.ndim - len(event_shape), values.ndim))
    return jnp.all(jnp.isfinite(values), axis=axes)


class BSDEPathBatch(StrictModule):
    """Forward states and aligned Wiener increments on one explicit time grid."""

    times: Array
    states: Array
    wiener_increments: Array
    valid: Array
    realization: StochasticRealization | None
    jump_events: frozendict[str, JumpEventBatch]
    metadata: frozendict[str, Any]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        wiener_increments: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int],
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        path_id: str,
        process_id: str,
        valid: ArrayLike | None = None,
        realization: StochasticRealization | None = None,
        jump_events: Mapping[str, JumpEventBatch] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        samples = _shape(sample_shape, owner="sample_shape") if sample_shape else ()
        state_event = _shape(state_shape, owner="state_shape")
        noise_event = _shape(noise_shape, owner="noise_shape")
        time_values = jnp.asarray(times, dtype=float)
        if time_values.ndim != 1 or time_values.shape[0] < 2:
            raise ValueError(
                "times must be a one-dimensional grid with at least two nodes."
            )
        if bool(jnp.any(~jnp.isfinite(time_values))) or bool(
            jnp.any(jnp.diff(time_values) <= 0.0)
        ):
            raise ValueError("times must be finite and strictly increasing.")
        state_values = jnp.asarray(states)
        increment_values = jnp.asarray(wiener_increments)
        expected_states = samples + (time_values.shape[0],) + state_event
        expected_increments = samples + (time_values.shape[0] - 1,) + noise_event
        if state_values.shape != expected_states:
            raise ValueError(
                f"states must have shape {expected_states}; got {state_values.shape}."
            )
        if increment_values.shape != expected_increments:
            raise ValueError(
                "wiener_increments must align with every path interval; "
                f"expected {expected_increments}, got {increment_values.shape}."
            )
        if valid is None:
            validity = _event_finite(state_values, state_event)
        else:
            validity = jnp.asarray(valid, dtype=bool)
            if validity.shape != samples + (time_values.shape[0],):
                raise ValueError("valid must have sample_shape + (num_nodes,) shape.")
        if realization is not None and not is_stochastic_realization(realization):
            raise TypeError(
                "realization must implement StochasticRealization or be None."
            )
        events = {} if jump_events is None else dict(jump_events)
        if any(not isinstance(value, JumpEventBatch) for value in events.values()):
            raise TypeError("jump_events values must be JumpEventBatch objects.")
        if any(not isinstance(label, str) or not label for label in events):
            raise ValueError("jump_events labels must be non-empty strings.")
        if any(value.batch_shape != samples for value in events.values()):
            raise ValueError("jump_events batch shapes must match sample_shape.")
        if any(value.state_shape != state_event for value in events.values()):
            raise ValueError("jump_events state shapes must match state_shape.")
        self.times = time_values
        self.states = state_values
        self.wiener_increments = increment_values
        self.valid = validity
        self.realization = realization
        self.jump_events = frozendict(events)
        self.metadata = frozendict({} if metadata is None else metadata)
        self.sample_shape = samples
        self.state_shape = state_event
        self.noise_shape = noise_event
        self.path_id = _name(path_id, owner="path_id")
        self.process_id = _name(process_id, owner="process_id")

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[0]) - 1

    @property
    def num_paths(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid, axis=-1)

    @property
    def path_valid(self) -> Array:
        """Per-path eligibility shared by BSDE objectives and integrations."""
        return self.successful


class BSDEProblem(StrictModule):
    """Markovian BSDE coefficients, terminal condition, and forward-path sampler."""

    forward_sampler: Callable[[Array], BSDEPathBatch]
    drift: Callable[[Array, Array, Any], Array]
    diffusion: Callable[[Array, Array, Any], Array]
    generator: Callable[[Array, Array, Array, Array, Any], Array]
    terminal: Callable[[Array, Any], Array]
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    state_label: str = eqx.field(static=True)

    def __init__(
        self,
        forward_sampler: Callable[[Array], BSDEPathBatch],
        drift: Callable[[Array, Array, Any], Array],
        diffusion: Callable[[Array, Array, Any], Array],
        generator: Callable[[Array, Array, Array, Array, Any], Array],
        terminal: Callable[[Array, Any], Array],
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        output_shape: Sequence[int],
        problem_id: str,
        process_id: str,
        args: Any = None,
        time_label: str = "t",
        state_label: str = "x",
    ):
        for owner, value in (
            ("forward_sampler", forward_sampler),
            ("drift", drift),
            ("diffusion", diffusion),
            ("generator", generator),
            ("terminal", terminal),
        ):
            if not callable(value):
                raise TypeError(f"{owner} must be callable.")
        labels = (
            _name(time_label, owner="time_label"),
            _name(state_label, owner="state_label"),
        )
        if labels[0] == labels[1]:
            raise ValueError("time_label and state_label must be distinct.")
        self.forward_sampler = forward_sampler
        self.drift = drift
        self.diffusion = diffusion
        self.generator = generator
        self.terminal = terminal
        self.args = args
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.noise_shape = _shape(noise_shape, owner="noise_shape")
        self.output_shape = _shape(output_shape, owner="output_shape")
        self.problem_id = _name(problem_id, owner="problem_id")
        self.process_id = _name(process_id, owner="process_id")
        self.time_label, self.state_label = labels

    def sample(self, key: Key[Array, ""], /) -> BSDEPathBatch:
        paths = self.forward_sampler(key)
        if not isinstance(paths, BSDEPathBatch):
            raise TypeError("forward_sampler must return a BSDEPathBatch.")
        if paths.state_shape != self.state_shape or paths.noise_shape != self.noise_shape:
            raise ValueError(
                "Forward path state/noise shapes do not match the BSDE problem."
            )
        if paths.process_id != self.process_id:
            raise ValueError("Forward path and BSDE process IDs do not match.")
        return paths


def bsde_paths_from_differential_solution(
    solution: Any,
    /,
    *,
    path_id: str,
    process_id: str,
    state_shape: Sequence[int] | None = None,
) -> BSDEPathBatch:
    """Recover aligned forward states and increments from a Wiener-driven solution."""
    realization = solution.realization
    if not isinstance(realization, WienerRealization):
        raise TypeError(
            "BSDE paths require a DifferentialSolution with WienerRealization."
        )
    solution_times = jnp.asarray(solution.times)
    if solution_times.ndim == 1:
        times = solution_times
    elif solution_times.shape[:-1] == solution.sample_shape:
        path_times = solution_times.reshape((-1, solution_times.shape[-1]))
        times = path_times[0]
        if not bool(jnp.all(path_times == times)):
            raise ValueError("All differential-solution paths must share one time grid.")
    else:
        raise ValueError(
            "Differential-solution times must have shape (num_nodes,) or "
            "sample_shape + (num_nodes,)."
        )
    increments = realization.increments(times[:-1], times[1:])
    resolved_state_shape = (
        tuple(solution.states.shape[len(solution.sample_shape) + 1 :])
        if state_shape is None
        else tuple(int(size) for size in state_shape)
    )
    return BSDEPathBatch(
        times,
        solution.states,
        increments,
        sample_shape=solution.sample_shape,
        state_shape=resolved_state_shape,
        noise_shape=realization.noise_shape,
        path_id=path_id,
        process_id=process_id,
        valid=solution.valid,
        realization=realization,
        metadata={
            "solver_name": solution.solver_name,
            "interpretation": solution.interpretation,
        },
    )


def _predictor_value(
    predictor: Callable | DomainFunction,
    time: Array,
    state: Array,
    problem: BSDEProblem,
    /,
    *,
    key: Array,
) -> Array:
    if isinstance(predictor, DomainFunction):
        arguments = []
        for dependency in predictor.deps:
            if dependency == problem.time_label:
                arguments.append(time)
            elif dependency == problem.state_label:
                arguments.append(state)
            else:
                raise ValueError(
                    f"BSDE predictor has unsupported dependency {dependency!r}."
                )
        value = predictor.func(*arguments, key=key)
    elif callable(predictor):
        value = predictor(time, state)
    else:
        raise TypeError("BSDE predictors must be callable or DomainFunction objects.")
    if isinstance(value, cx.Field):
        return jnp.asarray(value.data)
    return jnp.asarray(value)


def _pointwise_values(
    predictor: Callable | DomainFunction,
    times: Array,
    states: Array,
    problem: BSDEProblem,
    /,
    *,
    key: Array,
    output_shape: tuple[int, ...],
) -> Array:
    flat_states = states.reshape((-1,) + problem.state_shape)
    flat_times = jnp.broadcast_to(
        times, states.shape[: -len(problem.state_shape)]
    ).reshape((-1,))
    keys = jr.split(key, flat_states.shape[0])
    values = jax.vmap(
        lambda time, state, point_key: _predictor_value(
            predictor, time, state, problem, key=point_key
        )
    )(flat_times, flat_states, keys)
    expected = (flat_states.shape[0],) + output_shape
    if values.shape != expected:
        raise ValueError(
            f"BSDE predictor must return trailing shape {output_shape}; got {values.shape}."
        )
    return values.reshape(states.shape[: -len(problem.state_shape)] + output_shape)


def autodiff_bsde_control(
    value_predictor: Callable | DomainFunction,
    time: ArrayLike,
    state: ArrayLike,
    problem: BSDEProblem,
    /,
    *,
    key: Key[Array, ""] = jr.key(0),
) -> Array:
    """Compute Z = grad_x u sigma with full output/noise event semantics."""
    if not isinstance(problem, BSDEProblem):
        raise TypeError("problem must be a BSDEProblem.")
    time_value = jnp.asarray(time)
    state_value = jnp.asarray(state)
    if state_value.shape != problem.state_shape:
        raise ValueError("state must have exactly problem.state_shape.")

    def value(state_argument):
        return _predictor_value(
            value_predictor,
            time_value,
            state_argument,
            problem,
            key=key,
        )

    jacobian = jax.jacrev(value)(state_value)
    state_size = prod(problem.state_shape)
    output_size = prod(problem.output_shape)
    noise_size = prod(problem.noise_shape)
    jacobian_flat = jacobian.reshape((output_size, state_size))
    diffusion = jnp.asarray(problem.diffusion(time_value, state_value, problem.args))
    expected = problem.state_shape + problem.noise_shape
    if diffusion.shape != expected:
        raise ValueError(f"diffusion must have shape {expected}; got {diffusion.shape}.")
    control = jacobian_flat @ diffusion.reshape((state_size, noise_size))
    return control.reshape(problem.output_shape + problem.noise_shape)


def _pointwise_autodiff_control(
    predictor: Callable | DomainFunction,
    times: Array,
    states: Array,
    problem: BSDEProblem,
    /,
    *,
    key: Array,
) -> Array:
    flat_states = states.reshape((-1,) + problem.state_shape)
    flat_times = jnp.broadcast_to(
        times, states.shape[: -len(problem.state_shape)]
    ).reshape((-1,))
    keys = jr.split(key, flat_states.shape[0])
    values = jax.vmap(
        lambda time, state, point_key: autodiff_bsde_control(
            predictor, time, state, problem, key=point_key
        )
    )(flat_times, flat_states, keys)
    return values.reshape(
        states.shape[: -len(problem.state_shape)]
        + problem.output_shape
        + problem.noise_shape
    )


class BSDEEvaluation(StrictModule):
    """Terminal, local-increment, and global-trajectory residual decomposition."""

    values: Array
    controls: Array
    generator_values: Array
    terminal_residual: Array
    local_residuals: Array
    global_residual: Array
    martingale_increments: Array
    valid_paths: Array
    paths: BSDEPathBatch
    quadrature: BSDEQuadrature = eqx.field(static=True)
    control_mode: BSDEControlMode = eqx.field(static=True)


def evaluate_bsde(
    problem: BSDEProblem,
    paths: BSDEPathBatch,
    value_predictor: Callable | DomainFunction,
    /,
    *,
    control_predictor: Callable | DomainFunction | None = None,
    control_mode: BSDEControlMode = "explicit",
    quadrature: BSDEQuadrature = "left",
    key: Key[Array, ""] = jr.key(0),
) -> BSDEEvaluation:
    """Evaluate one Markovian BSDE on explicitly aligned path intervals."""
    if not isinstance(problem, BSDEProblem):
        raise TypeError("problem must be a BSDEProblem.")
    if not isinstance(paths, BSDEPathBatch):
        raise TypeError("paths must be a BSDEPathBatch.")
    if (
        paths.state_shape != problem.state_shape
        or paths.noise_shape != problem.noise_shape
    ):
        raise ValueError("Path and BSDE state/noise shapes do not match.")
    if control_mode not in ("explicit", "autodiff"):
        raise ValueError("control_mode must be 'explicit' or 'autodiff'.")
    if quadrature not in ("left", "trapezoid"):
        raise ValueError("quadrature must be 'left' or 'trapezoid'.")
    if control_mode == "explicit" and control_predictor is None:
        raise ValueError("Explicit BSDE control requires control_predictor.")
    value_key, control_key, right_control_key = jr.split(key, 3)
    values = _pointwise_values(
        value_predictor,
        paths.times,
        paths.states,
        problem,
        key=value_key,
        output_shape=problem.output_shape,
    )
    left_states = paths.states[..., :-1, *([slice(None)] * len(problem.state_shape))]
    right_states = paths.states[..., 1:, *([slice(None)] * len(problem.state_shape))]
    left_times = paths.times[:-1]
    right_times = paths.times[1:]
    generator_states = left_states
    generator_times = left_times
    if control_mode == "autodiff":
        controls = _pointwise_autodiff_control(
            value_predictor,
            generator_times,
            generator_states,
            problem,
            key=control_key,
        )
    else:
        if control_predictor is None:
            raise RuntimeError("Explicit BSDE control predictor is unavailable.")
        controls = _pointwise_values(
            control_predictor,
            generator_times,
            generator_states,
            problem,
            key=control_key,
            output_shape=problem.output_shape + problem.noise_shape,
        )
    generator_y = values[..., :-1, *([slice(None)] * len(problem.output_shape))]
    sample_count = prod(paths.sample_shape) if paths.sample_shape else 1
    state_size = prod(problem.state_shape)
    output_size = prod(problem.output_shape)
    noise_size = prod(problem.noise_shape)

    def generator_value(time, state, value, control):
        output = jnp.asarray(problem.generator(time, state, value, control, problem.args))
        if output.shape != problem.output_shape:
            raise ValueError("BSDE generator returned an incompatible output shape.")
        return output

    flat_times = jnp.broadcast_to(
        generator_times,
        paths.sample_shape + (paths.num_steps,),
    ).reshape((-1,))
    flat_states = generator_states.reshape((-1,) + problem.state_shape)
    flat_y = generator_y.reshape((-1,) + problem.output_shape)
    flat_z = controls.reshape((-1,) + problem.output_shape + problem.noise_shape)
    generator_values = jax.vmap(generator_value)(
        flat_times, flat_states, flat_y, flat_z
    ).reshape(paths.sample_shape + (paths.num_steps,) + problem.output_shape)
    if quadrature == "trapezoid":
        if control_mode == "autodiff":
            right_controls = _pointwise_autodiff_control(
                value_predictor,
                right_times,
                right_states,
                problem,
                key=right_control_key,
            )
        else:
            if control_predictor is None:
                raise RuntimeError("Explicit BSDE control predictor is unavailable.")
            right_controls = _pointwise_values(
                control_predictor,
                right_times,
                right_states,
                problem,
                key=right_control_key,
                output_shape=problem.output_shape + problem.noise_shape,
            )
        flat_right_y = values[
            ..., 1:, *([slice(None)] * len(problem.output_shape))
        ].reshape((-1,) + problem.output_shape)
        flat_right_z = right_controls.reshape(
            (-1,) + problem.output_shape + problem.noise_shape
        )
        right_generator = jax.vmap(generator_value)(
            jnp.broadcast_to(
                right_times, paths.sample_shape + (paths.num_steps,)
            ).reshape((-1,)),
            right_states.reshape((-1,) + problem.state_shape),
            flat_right_y,
            flat_right_z,
        ).reshape(paths.sample_shape + (paths.num_steps,) + problem.output_shape)
        generator_values = 0.5 * (generator_values + right_generator)
    controls_flat = controls.reshape(
        (sample_count, paths.num_steps, output_size, noise_size)
    )
    increments_flat = paths.wiener_increments.reshape(
        (sample_count, paths.num_steps, noise_size)
    )
    martingale = oe.contract("ston,stn->sto", controls_flat, increments_flat).reshape(
        paths.sample_shape + (paths.num_steps,) + problem.output_shape
    )
    dt = jnp.diff(paths.times)
    drift_increment = generator_values * dt.reshape(
        (1,) * len(paths.sample_shape)
        + (paths.num_steps,)
        + (1,) * len(problem.output_shape)
    )
    value_increment = (
        values[..., 1:, *([slice(None)] * len(problem.output_shape))]
        - values[..., :-1, *([slice(None)] * len(problem.output_shape))]
    )
    local_residuals = value_increment + drift_increment - martingale
    terminal_states = paths.states[..., -1, *([slice(None)] * len(problem.state_shape))]
    terminal_target = jax.vmap(
        lambda state: jnp.asarray(problem.terminal(state, problem.args))
    )(terminal_states.reshape((-1,) + problem.state_shape)).reshape(
        paths.sample_shape + problem.output_shape
    )
    terminal_residual = (
        values[..., -1, *([slice(None)] * len(problem.output_shape))] - terminal_target
    )
    global_residual = (
        values[..., -1, *([slice(None)] * len(problem.output_shape))]
        - values[..., 0, *([slice(None)] * len(problem.output_shape))]
        + jnp.sum(drift_increment, axis=len(paths.sample_shape))
        - jnp.sum(martingale, axis=len(paths.sample_shape))
    )
    interval_valid = paths.valid[..., :-1] & paths.valid[..., 1:]
    finite = _event_finite(local_residuals, problem.output_shape) & _event_finite(
        martingale, problem.output_shape
    )
    valid_paths = (
        paths.path_valid
        & jnp.all(interval_valid & finite, axis=-1)
        & _event_finite(terminal_residual, problem.output_shape)
    )
    return BSDEEvaluation(
        values=values,
        controls=controls,
        generator_values=generator_values,
        terminal_residual=terminal_residual,
        local_residuals=local_residuals,
        global_residual=global_residual,
        martingale_increments=martingale,
        valid_paths=valid_paths,
        paths=paths,
        quadrature=quadrature,
        control_mode=control_mode,
    )


def _masked_mean_square(
    values: Array, valid: Array, event_shape: tuple[int, ...], /
) -> Array:
    squared = jnp.abs(values) ** 2
    if event_shape:
        squared = jnp.sum(
            squared,
            axis=tuple(range(squared.ndim - len(event_shape), squared.ndim)),
        )
    mask = jnp.broadcast_to(valid, squared.shape)
    return jnp.sum(jnp.where(mask, squared, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def bsde_objective_loss(
    evaluation: BSDEEvaluation,
    /,
    *,
    mode: BSDEObjectiveMode = "joint",
    terminal_weight: ArrayLike = 1.0,
    local_weight: ArrayLike = 1.0,
    global_weight: ArrayLike = 1.0,
) -> Array:
    """Compose terminal, local, and global residual losses without hidden terms."""
    if not isinstance(evaluation, BSDEEvaluation):
        raise TypeError("evaluation must be a BSDEEvaluation.")
    if mode not in ("terminal", "local", "global", "joint"):
        raise ValueError("Unknown BSDE objective mode.")
    weights = tuple(
        jnp.asarray(value, dtype=float).reshape(())
        for value in (terminal_weight, local_weight, global_weight)
    )
    if any(bool(~jnp.isfinite(value)) or float(value) < 0.0 for value in weights):
        raise ValueError("BSDE objective weights must be finite and nonnegative.")
    problem_output_shape = evaluation.terminal_residual.shape[
        len(evaluation.paths.sample_shape) :
    ]
    terminal_loss = _masked_mean_square(
        evaluation.terminal_residual,
        evaluation.valid_paths,
        problem_output_shape,
    )
    interval_valid = (
        evaluation.paths.valid[..., :-1]
        & evaluation.paths.valid[..., 1:]
        & evaluation.valid_paths[..., None]
    )
    local_loss = _masked_mean_square(
        evaluation.local_residuals,
        interval_valid,
        problem_output_shape,
    )
    global_loss = _masked_mean_square(
        evaluation.global_residual,
        evaluation.valid_paths,
        problem_output_shape,
    )
    terminal_weight_value, local_weight_value, global_weight_value = weights
    if mode == "terminal":
        return terminal_weight_value * terminal_loss
    if mode == "local":
        return terminal_weight_value * terminal_loss + local_weight_value * local_loss
    if mode == "global":
        return terminal_weight_value * terminal_loss + global_weight_value * global_loss
    return (
        terminal_weight_value * terminal_loss
        + local_weight_value * local_loss
        + global_weight_value * global_loss
    )


class BSDEDiagnostics(StrictModule):
    """Residual scales, martingale centering, and path-validity diagnostics."""

    terminal_rmse: Array
    local_rmse: Array
    global_rmse: Array
    martingale_mean: Array
    valid_fraction: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.finite)) and bool(self.valid_fraction > 0.0)


def bsde_diagnostics(evaluation: BSDEEvaluation, /) -> BSDEDiagnostics:
    if not isinstance(evaluation, BSDEEvaluation):
        raise TypeError("evaluation must be a BSDEEvaluation.")
    sample_axes = tuple(range(len(evaluation.paths.sample_shape)))
    local_axes = sample_axes + (len(evaluation.paths.sample_shape),)
    terminal_rmse = jnp.sqrt(
        jnp.mean(jnp.abs(evaluation.terminal_residual) ** 2, axis=sample_axes)
    )
    local_rmse = jnp.sqrt(
        jnp.mean(jnp.abs(evaluation.local_residuals) ** 2, axis=local_axes)
    )
    global_rmse = jnp.sqrt(
        jnp.mean(jnp.abs(evaluation.global_residual) ** 2, axis=sample_axes)
    )
    martingale_mean = jnp.mean(evaluation.martingale_increments, axis=local_axes)
    finite = jnp.asarray(
        jnp.all(jnp.isfinite(terminal_rmse))
        & jnp.all(jnp.isfinite(local_rmse))
        & jnp.all(jnp.isfinite(global_rmse))
        & jnp.all(jnp.isfinite(martingale_mean))
    )
    return BSDEDiagnostics(
        terminal_rmse=terminal_rmse,
        local_rmse=local_rmse,
        global_rmse=global_rmse,
        martingale_mean=martingale_mean,
        valid_fraction=jnp.mean(evaluation.valid_paths),
        finite=finite,
    )


def semilinear_pde_residual(
    problem: BSDEProblem,
    value_predictor: Callable | DomainFunction,
    time: ArrayLike,
    state: ArrayLike,
    /,
    *,
    key: Key[Array, ""] = jr.key(0),
) -> Array:
    """Evaluate u_t + b·grad u + 1/2 tr(sigma sigmaᵀ Hess u) + f(t,x,u,Z)."""
    if not isinstance(problem, BSDEProblem):
        raise TypeError("problem must be a BSDEProblem.")
    time_value = jnp.asarray(time)
    state_value = jnp.asarray(state)
    if time_value.shape != () or state_value.shape != problem.state_shape:
        raise ValueError("time must be scalar and state must equal problem.state_shape.")

    def value_at_time(time_argument):
        return _predictor_value(
            value_predictor, time_argument, state_value, problem, key=key
        )

    def value_at_state(state_argument):
        return _predictor_value(
            value_predictor, time_value, state_argument, problem, key=key
        )

    value = value_at_state(state_value)
    time_derivative = jax.jacrev(value_at_time)(time_value)
    jacobian = jax.jacrev(value_at_state)(state_value)
    hessian = jax.jacfwd(jax.jacrev(value_at_state))(state_value)
    state_size = prod(problem.state_shape)
    output_size = prod(problem.output_shape)
    noise_size = prod(problem.noise_shape)
    drift = jnp.asarray(problem.drift(time_value, state_value, problem.args)).reshape(
        (state_size,)
    )
    diffusion = jnp.asarray(
        problem.diffusion(time_value, state_value, problem.args)
    ).reshape((state_size, noise_size))
    jacobian_flat = jacobian.reshape((output_size, state_size))
    hessian_flat = hessian.reshape((output_size, state_size, state_size))
    covariance = diffusion @ diffusion.T
    generator_action = (
        jacobian_flat @ drift + 0.5 * oe.contract("ij,oij->o", covariance, hessian_flat)
    ).reshape(problem.output_shape)
    control = (jacobian_flat @ diffusion).reshape(
        problem.output_shape + problem.noise_shape
    )
    nonlinear = jnp.asarray(
        problem.generator(time_value, state_value, value, control, problem.args)
    )
    if nonlinear.shape != problem.output_shape:
        raise ValueError("BSDE generator returned an incompatible output shape.")
    return time_derivative + generator_action + nonlinear


__all__ = [
    "autodiff_bsde_control",
    "bsde_diagnostics",
    "BSDEControlMode",
    "BSDEDiagnostics",
    "BSDEEvaluation",
    "bsde_objective_loss",
    "BSDEObjectiveMode",
    "BSDEPathBatch",
    "bsde_paths_from_differential_solution",
    "BSDEProblem",
    "BSDEQuadrature",
    "evaluate_bsde",
    "semilinear_pde_residual",
]
