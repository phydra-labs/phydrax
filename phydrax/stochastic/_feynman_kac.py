#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain._function import DomainFunction
from ._bsde import (
    _event_finite,
    _predictor_value,
    autodiff_bsde_control,
    BSDEPathBatch,
    BSDEProblem,
    BSDEQuadrature,
)


FeynmanKacSamplingMode: TypeAlias = Literal["trajectory_nodes", "queries"]
FeynmanKacControlTargetMode: TypeAlias = Literal["none", "martingale", "malliavin"]
FeynmanKacRefreshMode: TypeAlias = Literal["fixed", "resample"]
FeynmanKacTimeWeighting: TypeAlias = Literal["uniform", "trapezoid"]
Predictor: TypeAlias = Callable | DomainFunction
MalliavinWeight: TypeAlias = Callable[[Array, Array, Array, Array, Any], Array]


def _positive_shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return shape


def _event_axes(ndim: int, event_shape: tuple[int, ...], /) -> tuple[int, ...]:
    return tuple(range(ndim - len(event_shape), ndim))


def _plan_id(parts: tuple[Any, ...], /) -> str:
    digest = sha256(b"phydrax-feynman-kac-plan\0")
    digest.update(repr(parts).encode("utf-8"))
    return digest.hexdigest()


class FeynmanKacSamplingPlan(StrictModule):
    """Static numerical policy for global-in-time Feynman--Kac labels."""

    terminal_time: float = eqx.field(static=True)
    initial_time: float = eqx.field(static=True)
    sampling_mode: FeynmanKacSamplingMode = eqx.field(static=True)
    num_paths_per_query: int = eqx.field(static=True)
    num_time_steps: int = eqx.field(static=True)
    quadrature: BSDEQuadrature = eqx.field(static=True)
    control_target_mode: FeynmanKacControlTargetMode = eqx.field(static=True)
    antithetic: bool = eqx.field(static=True)
    path_chunk_size: int | None = eqx.field(static=True)
    time_weighting: FeynmanKacTimeWeighting = eqx.field(static=True)
    refresh_mode: FeynmanKacRefreshMode = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        terminal_time: float,
        initial_time: float = 0.0,
        sampling_mode: FeynmanKacSamplingMode = "trajectory_nodes",
        num_paths_per_query: int = 1,
        num_time_steps: int = 32,
        quadrature: BSDEQuadrature = "left",
        control_target_mode: FeynmanKacControlTargetMode = "none",
        antithetic: bool = False,
        path_chunk_size: int | None = None,
        time_weighting: FeynmanKacTimeWeighting = "uniform",
        refresh_mode: FeynmanKacRefreshMode = "resample",
        plan_id: str | None = None,
    ):
        t0 = float(initial_time)
        t1 = float(terminal_time)
        if not jnp.isfinite(t0) or not jnp.isfinite(t1) or t1 <= t0:
            raise ValueError("initial_time and terminal_time must be finite with t1 > t0.")
        if sampling_mode not in ("trajectory_nodes", "queries"):
            raise ValueError("Unknown Feynman-Kac sampling_mode.")
        paths = int(num_paths_per_query)
        steps = int(num_time_steps)
        if paths < 1 or steps < 1:
            raise ValueError("num_paths_per_query and num_time_steps must be positive.")
        if quadrature not in ("left", "trapezoid"):
            raise ValueError("quadrature must be 'left' or 'trapezoid'.")
        if control_target_mode not in ("none", "martingale", "malliavin"):
            raise ValueError("Unknown Feynman-Kac control_target_mode.")
        use_antithetic = bool(antithetic)
        if use_antithetic and paths % 2:
            raise ValueError("Antithetic sampling requires an even path count.")
        chunk = None if path_chunk_size is None else int(path_chunk_size)
        if chunk is not None and (chunk < 1 or chunk > paths):
            raise ValueError("path_chunk_size must lie in [1, num_paths_per_query].")
        if use_antithetic and chunk is not None and chunk % 2:
            raise ValueError("Antithetic path chunks must contain an even path count.")
        if time_weighting not in ("uniform", "trapezoid"):
            raise ValueError("time_weighting must be 'uniform' or 'trapezoid'.")
        if refresh_mode not in ("fixed", "resample"):
            raise ValueError("refresh_mode must be 'fixed' or 'resample'.")
        identity_parts = (
            t0,
            t1,
            sampling_mode,
            paths,
            steps,
            quadrature,
            control_target_mode,
            use_antithetic,
            chunk,
            time_weighting,
            refresh_mode,
        )
        self.initial_time = t0
        self.terminal_time = t1
        self.sampling_mode = sampling_mode
        self.num_paths_per_query = paths
        self.num_time_steps = steps
        self.quadrature = quadrature
        self.control_target_mode = control_target_mode
        self.antithetic = use_antithetic
        self.path_chunk_size = chunk
        self.time_weighting = time_weighting
        self.refresh_mode = refresh_mode
        self.plan_id = _plan_id(identity_parts) if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be non-empty.")


class FeynmanKacPathBatch(StrictModule):
    """Query-conditioned continuation paths on normalized per-query time grids."""

    query_times: Array
    query_states: Array
    query_weights: Array
    times: Array
    states: Array
    wiener_increments: Array
    valid: Array
    dependence_ids: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    antithetic: bool = eqx.field(static=True)

    def __init__(
        self,
        query_times: ArrayLike,
        query_states: ArrayLike,
        times: ArrayLike,
        states: ArrayLike,
        wiener_increments: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        path_id: str,
        process_id: str,
        valid: ArrayLike | None = None,
        query_weights: ArrayLike | None = None,
        dependence_ids: ArrayLike | None = None,
        antithetic: bool = False,
    ):
        state_event = _positive_shape(state_shape, owner="state_shape")
        noise_event = _positive_shape(noise_shape, owner="noise_shape")
        q_times = jnp.asarray(query_times, dtype=float).reshape((-1,))
        q_states = jnp.asarray(query_states)
        if q_states.shape != (q_times.shape[0],) + state_event:
            raise ValueError("query_states must have shape (num_queries,) + state_shape.")
        time_values = jnp.asarray(times, dtype=float)
        if time_values.ndim != 2 or time_values.shape[0] != q_times.shape[0]:
            raise ValueError("times must have shape (num_queries, num_steps + 1).")
        if time_values.shape[1] < 2:
            raise ValueError("Continuation paths require at least one time step.")
        state_values = jnp.asarray(states)
        increment_values = jnp.asarray(wiener_increments)
        num_paths = int(state_values.shape[1]) if state_values.ndim >= 3 else 0
        num_steps = int(time_values.shape[1]) - 1
        expected_states = (q_times.shape[0], num_paths, num_steps + 1) + state_event
        expected_increments = (q_times.shape[0], num_paths, num_steps) + noise_event
        if state_values.shape != expected_states:
            raise ValueError(f"states must have shape {expected_states}.")
        if increment_values.shape != expected_increments:
            raise ValueError(f"wiener_increments must have shape {expected_increments}.")
        if num_paths < 1:
            raise ValueError("Continuation batches require at least one path per query.")
        if valid is None:
            validity = _event_finite(state_values, state_event)
        else:
            validity = jnp.asarray(valid, dtype=bool)
            if validity.shape != expected_states[:3]:
                raise ValueError("valid must have shape (query, path, time).")
        if query_weights is None:
            weights = jnp.ones((q_times.shape[0],), dtype=float)
        else:
            weights = jnp.asarray(query_weights, dtype=float).reshape((-1,))
            if weights.shape != q_times.shape:
                raise ValueError("query_weights must align with query_times.")
        if bool(jnp.any(~jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
            raise ValueError("query_weights must be finite and nonnegative.")
        if dependence_ids is None:
            ids = jnp.broadcast_to(
                jnp.arange(num_paths, dtype=jnp.int32),
                (q_times.shape[0], num_paths),
            )
        else:
            ids = jnp.asarray(dependence_ids, dtype=jnp.int32)
            if ids.shape != (q_times.shape[0], num_paths):
                raise ValueError("dependence_ids must have shape (query, path).")
        self.query_times = q_times
        self.query_states = q_states
        self.query_weights = weights
        self.times = time_values
        self.states = state_values
        self.wiener_increments = increment_values
        self.valid = validity
        self.dependence_ids = ids
        self.state_shape = state_event
        self.noise_shape = noise_event
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.path_id = str(path_id)
        self.process_id = str(process_id)
        self.antithetic = bool(antithetic)
        if not self.path_id or not self.process_id:
            raise ValueError("path_id and process_id must be non-empty.")


class FeynmanKacLabelBatch(StrictModule):
    """Value and optional control regression targets with dependence metadata."""

    query_times: Array
    query_states: Array
    value_targets: Array
    value_standard_errors: Array
    control_targets: Array | None
    control_standard_errors: Array | None
    valid: Array
    control_valid: Array
    sample_weights: Array
    cluster_ids: Array
    metadata: frozendict[str, Any]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    source_path_count: int = eqx.field(static=True)

    def __init__(
        self,
        query_times: ArrayLike,
        query_states: ArrayLike,
        value_targets: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        output_shape: Sequence[int],
        problem_id: str,
        process_id: str,
        plan_id: str,
        value_standard_errors: ArrayLike | None = None,
        control_targets: ArrayLike | None = None,
        control_standard_errors: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        control_valid: ArrayLike | None = None,
        sample_weights: ArrayLike | None = None,
        cluster_ids: ArrayLike | None = None,
        source_path_count: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ):
        state_event = _positive_shape(state_shape, owner="state_shape")
        noise_event = _positive_shape(noise_shape, owner="noise_shape")
        output_event = _positive_shape(output_shape, owner="output_shape")
        times = jnp.asarray(query_times, dtype=float).reshape((-1,))
        count = int(times.shape[0])
        states = jnp.asarray(query_states)
        targets = jnp.asarray(value_targets)
        if states.shape != (count,) + state_event:
            raise ValueError("query_states must have shape (count,) + state_shape.")
        if targets.shape != (count,) + output_event:
            raise ValueError("value_targets must have shape (count,) + output_shape.")
        if value_standard_errors is None:
            value_errors = jnp.full(targets.shape, jnp.nan, dtype=float)
        else:
            value_errors = jnp.asarray(value_standard_errors, dtype=float)
            if value_errors.shape != targets.shape:
                raise ValueError("value_standard_errors must match value_targets.")
        controls = None if control_targets is None else jnp.asarray(control_targets)
        control_errors = (
            None
            if control_standard_errors is None
            else jnp.asarray(control_standard_errors, dtype=float)
        )
        control_shape = (count,) + output_event + noise_event
        if controls is not None and controls.shape != control_shape:
            raise ValueError(f"control_targets must have shape {control_shape}.")
        if control_errors is not None and (
            controls is None or control_errors.shape != controls.shape
        ):
            raise ValueError("control_standard_errors must match control_targets.")
        if valid is None:
            validity = jnp.isfinite(times) & _event_finite(states, state_event)
            validity = validity & _event_finite(targets, output_event)
        else:
            validity = jnp.asarray(valid, dtype=bool).reshape((-1,))
            if validity.shape != times.shape:
                raise ValueError("valid must align with query_times.")
        if control_valid is None:
            control_validity = validity if controls is not None else jnp.zeros_like(validity)
        else:
            control_validity = jnp.asarray(control_valid, dtype=bool).reshape((-1,))
            if control_validity.shape != times.shape:
                raise ValueError("control_valid must align with query_times.")
        if controls is not None:
            control_validity = control_validity & _event_finite(
                controls, output_event + noise_event
            )
        if sample_weights is None:
            weights = jnp.ones((count,), dtype=float)
        else:
            weights = jnp.asarray(sample_weights, dtype=float).reshape((-1,))
            if weights.shape != times.shape:
                raise ValueError("sample_weights must align with query_times.")
        if bool(jnp.any(~jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
            raise ValueError("sample_weights must be finite and nonnegative.")
        ids = (
            jnp.arange(count, dtype=jnp.int32)
            if cluster_ids is None
            else jnp.asarray(cluster_ids, dtype=jnp.int32).reshape((-1,))
        )
        if ids.shape != times.shape:
            raise ValueError("cluster_ids must align with query_times.")
        source_count = int(source_path_count)
        if source_count < 1:
            raise ValueError("source_path_count must be positive.")
        self.query_times = times
        self.query_states = states
        self.value_targets = targets
        self.value_standard_errors = value_errors
        self.control_targets = controls
        self.control_standard_errors = control_errors
        self.valid = validity
        self.control_valid = control_validity
        self.sample_weights = weights
        self.cluster_ids = ids
        self.metadata = frozendict({} if metadata is None else metadata)
        self.state_shape = state_event
        self.noise_shape = noise_event
        self.output_shape = output_event
        self.problem_id = str(problem_id)
        self.process_id = str(process_id)
        self.plan_id = str(plan_id)
        self.source_path_count = source_count
        if not self.problem_id or not self.process_id or not self.plan_id:
            raise ValueError("problem_id, process_id, and plan_id must be non-empty.")

    @property
    def num_queries(self) -> int:
        return int(self.query_times.shape[0])


class FeynmanKacLabelDiagnostics(StrictModule):
    valid_fraction: Array
    control_valid_fraction: Array
    mean_value_standard_error: Array
    mean_control_standard_error: Array
    effective_sample_size: Array
    time_min: Array
    time_max: Array
    cluster_count: Array
    source_path_count: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


def feynman_kac_label_diagnostics(
    batch: FeynmanKacLabelBatch, /
) -> FeynmanKacLabelDiagnostics:
    if not isinstance(batch, FeynmanKacLabelBatch):
        raise TypeError("batch must be a FeynmanKacLabelBatch.")
    valid_weights = jnp.where(batch.valid, batch.sample_weights, 0.0)
    weight_sum = jnp.sum(valid_weights)
    weight_square_sum = jnp.sum(valid_weights**2)
    ess = jnp.where(weight_square_sum > 0.0, weight_sum**2 / weight_square_sum, 0.0)
    value_axes = _event_axes(
        batch.value_standard_errors.ndim,
        batch.output_shape,
    )
    finite_value_errors = jnp.where(
        batch.valid
        & jnp.all(jnp.isfinite(batch.value_standard_errors), axis=value_axes),
        jnp.mean(batch.value_standard_errors, axis=value_axes),
        jnp.nan,
    )
    mean_value_error = jnp.nanmean(finite_value_errors)
    if batch.control_standard_errors is None:
        mean_control_error = jnp.asarray(jnp.nan)
    else:
        control_axes = _event_axes(
            batch.control_standard_errors.ndim,
            batch.output_shape + batch.noise_shape,
        )
        finite_control_errors = jnp.where(
            batch.control_valid
            & jnp.all(jnp.isfinite(batch.control_standard_errors), axis=control_axes),
            jnp.mean(batch.control_standard_errors, axis=control_axes),
            jnp.nan,
        )
        mean_control_error = jnp.nanmean(finite_control_errors)
    valid_times = jnp.where(batch.valid, batch.query_times, jnp.nan)
    time_min = jnp.nanmin(valid_times)
    time_max = jnp.nanmax(valid_times)
    flat_targets = batch.value_targets.reshape((batch.num_queries, -1))
    finite_targets = jnp.where(batch.valid[..., None], flat_targets, 0.0)
    finite = (
        jnp.all(jnp.isfinite(finite_targets))
        & jnp.isfinite(weight_sum)
        & (weight_sum > 0.0)
    )
    return FeynmanKacLabelDiagnostics(
        valid_fraction=jnp.mean(batch.valid),
        control_valid_fraction=jnp.mean(batch.control_valid),
        mean_value_standard_error=mean_value_error,
        mean_control_standard_error=mean_control_error,
        effective_sample_size=ess,
        time_min=time_min,
        time_max=time_max,
        cluster_count=jnp.asarray(jnp.unique(batch.cluster_ids).shape[0]),
        source_path_count=jnp.asarray(batch.source_path_count),
        finite=jnp.asarray(finite),
    )


def _point_values(
    predictor: Predictor,
    times: Array,
    states: Array,
    problem: BSDEProblem,
    /,
    *,
    key: Key[Array, ""],
    output_shape: tuple[int, ...],
) -> Array:
    leading_shape = states.shape[: -len(problem.state_shape)]
    flat_states = states.reshape((-1,) + problem.state_shape)
    flat_times = jnp.broadcast_to(times, leading_shape).reshape((-1,))
    keys = jr.split(key, flat_states.shape[0])
    values = jax.vmap(
        lambda time, state, point_key: _predictor_value(
            predictor,
            time,
            state,
            problem,
            key=point_key,
        )
    )(flat_times, flat_states, keys)
    expected = (flat_states.shape[0],) + output_shape
    if values.shape != expected:
        raise ValueError(
            f"Predictor must return trailing shape {output_shape}; got {values.shape}."
        )
    return values.reshape(leading_shape + output_shape)


def _point_controls(
    value_predictor: Predictor,
    times: Array,
    states: Array,
    problem: BSDEProblem,
    /,
    *,
    key: Key[Array, ""],
) -> Array:
    leading_shape = states.shape[: -len(problem.state_shape)]
    flat_states = states.reshape((-1,) + problem.state_shape)
    flat_times = jnp.broadcast_to(times, leading_shape).reshape((-1,))
    keys = jr.split(key, flat_states.shape[0])
    controls = jax.vmap(
        lambda time, state, point_key: autodiff_bsde_control(
            value_predictor,
            time,
            state,
            problem,
            key=point_key,
        )
    )(flat_times, flat_states, keys)
    return controls.reshape(leading_shape + problem.output_shape + problem.noise_shape)


def _source_generator_nodes(
    problem: BSDEProblem,
    times: Array,
    states: Array,
    /,
    *,
    source_value: Predictor | None,
    source_control: Predictor | None,
    key: Key[Array, ""],
) -> Array:
    value_key, control_key = jr.split(key)
    leading_shape = states.shape[: -len(problem.state_shape)]
    if source_value is None:
        values = jnp.zeros(leading_shape + problem.output_shape, dtype=states.dtype)
    else:
        values = _point_values(
            source_value,
            times,
            states,
            problem,
            key=value_key,
            output_shape=problem.output_shape,
        )
    if source_control is not None:
        controls = _point_values(
            source_control,
            times,
            states,
            problem,
            key=control_key,
            output_shape=problem.output_shape + problem.noise_shape,
        )
    elif source_value is not None:
        controls = _point_controls(
            source_value,
            times,
            states,
            problem,
            key=control_key,
        )
    else:
        controls = jnp.zeros(
            leading_shape + problem.output_shape + problem.noise_shape,
            dtype=states.dtype,
        )
    flat_times = jnp.broadcast_to(times, leading_shape).reshape((-1,))
    flat_states = states.reshape((-1,) + problem.state_shape)
    flat_values = values.reshape((-1,) + problem.output_shape)
    flat_controls = controls.reshape(
        (-1,) + problem.output_shape + problem.noise_shape
    )

    def evaluate(time, state, value, control):
        result = jnp.asarray(problem.generator(time, state, value, control, problem.args))
        if result.shape != problem.output_shape:
            raise ValueError("BSDE generator returned an incompatible output shape.")
        return result

    return jax.vmap(evaluate)(
        flat_times,
        flat_states,
        flat_values,
        flat_controls,
    ).reshape(leading_shape + problem.output_shape)


def _terminal_values(problem: BSDEProblem, states: Array, /) -> Array:
    leading_shape = states.shape[: -len(problem.state_shape)]
    values = jax.vmap(
        lambda state: jnp.asarray(problem.terminal(state, problem.args))
    )(states.reshape((-1,) + problem.state_shape))
    expected = (prod(leading_shape),) + problem.output_shape
    if values.shape != expected:
        raise ValueError(
            f"Terminal function must return shape {problem.output_shape}; got {values.shape}."
        )
    return values.reshape(leading_shape + problem.output_shape)


def _reverse_targets(
    terminal: Array,
    generator_nodes: Array,
    times: Array,
    /,
    *,
    time_axis: int,
    quadrature: BSDEQuadrature,
) -> Array:
    left = jnp.take(generator_nodes, jnp.arange(generator_nodes.shape[time_axis] - 1), axis=time_axis)
    if quadrature == "trapezoid":
        right = jnp.take(generator_nodes, jnp.arange(1, generator_nodes.shape[time_axis]), axis=time_axis)
        interval_values = 0.5 * (left + right)
    else:
        interval_values = left
    dt = jnp.diff(times, axis=-1)
    while dt.ndim < interval_values.ndim - len(terminal.shape[time_axis:]):
        dt = jnp.expand_dims(dt, axis=-2)
    event_rank = interval_values.ndim - time_axis - 1
    dt_shape = dt.shape + (1,) * event_rank
    increments = interval_values * dt.reshape(dt_shape)
    reversed_sum = jnp.flip(
        jnp.cumsum(jnp.flip(increments, axis=time_axis), axis=time_axis),
        axis=time_axis,
    )
    expanded_terminal = jnp.expand_dims(terminal, axis=time_axis)
    interior = reversed_sum + expanded_terminal
    return jnp.concatenate((interior, expanded_terminal), axis=time_axis)


def _continuation_valid(valid: Array, /, *, time_axis: int) -> Array:
    return jnp.flip(
        jnp.cumprod(jnp.flip(valid, axis=time_axis), axis=time_axis).astype(bool),
        axis=time_axis,
    )


def _node_weights(times: Array, mode: FeynmanKacTimeWeighting, /) -> Array:
    if mode == "uniform":
        return jnp.ones_like(times)
    dt = jnp.diff(times)
    return jnp.concatenate(
        (
            0.5 * dt[:1],
            0.5 * (dt[:-1] + dt[1:]),
            0.5 * dt[-1:],
        )
    )


def trajectory_node_feynman_kac_labels(
    problem: BSDEProblem,
    paths: BSDEPathBatch,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    source_value: Predictor | None = None,
    source_control: Predictor | None = None,
    key: Key[Array, ""] = jr.key(0),
) -> FeynmanKacLabelBatch:
    """Construct one correlated global-time label at every valid trajectory node."""
    if not isinstance(problem, BSDEProblem) or not isinstance(paths, BSDEPathBatch):
        raise TypeError("problem and paths must be BSDEProblem and BSDEPathBatch objects.")
    if not isinstance(plan, FeynmanKacSamplingPlan):
        raise TypeError("plan must be a FeynmanKacSamplingPlan.")
    if plan.sampling_mode != "trajectory_nodes":
        raise ValueError("trajectory_node_feynman_kac_labels requires trajectory_nodes mode.")
    if paths.state_shape != problem.state_shape or paths.noise_shape != problem.noise_shape:
        raise ValueError("Path and problem state/noise shapes do not match.")
    if paths.process_id != problem.process_id:
        raise ValueError("Path and problem process IDs do not match.")
    if not bool(jnp.isclose(paths.times[-1], plan.terminal_time)):
        raise ValueError("Path terminal time does not match the sampling plan.")
    source_key = jr.fold_in(key, 1)
    time_axis = len(paths.sample_shape)
    generator = _source_generator_nodes(
        problem,
        paths.times,
        paths.states,
        source_value=source_value,
        source_control=source_control,
        key=source_key,
    )
    terminal_states = jnp.take(paths.states, -1, axis=time_axis)
    terminal = _terminal_values(problem, terminal_states)
    targets = _reverse_targets(
        terminal,
        generator,
        paths.times,
        time_axis=time_axis,
        quadrature=plan.quadrature,
    )
    continuation_valid = _continuation_valid(paths.valid, time_axis=time_axis)
    nodes = int(paths.times.shape[0])
    path_count = paths.num_paths
    query_times = jnp.broadcast_to(paths.times, paths.sample_shape + (nodes,)).reshape((-1,))
    query_states = paths.states.reshape((-1,) + problem.state_shape)
    value_targets = targets.reshape((-1,) + problem.output_shape)
    valid = continuation_valid.reshape((-1,)) & _event_finite(
        value_targets, problem.output_shape
    )
    node_weights = _node_weights(paths.times, plan.time_weighting)
    sample_weights = jnp.broadcast_to(
        node_weights,
        paths.sample_shape + (nodes,),
    ).reshape((-1,))
    cluster_ids = jnp.repeat(jnp.arange(path_count, dtype=jnp.int32), nodes)
    control_targets = None
    control_errors = None
    control_valid = jnp.zeros_like(valid)
    if plan.control_target_mode == "martingale":
        output_size = prod(problem.output_shape)
        noise_size = prod(problem.noise_shape)
        next_values = jnp.take(targets, jnp.arange(1, nodes), axis=time_axis)
        target_flat = next_values.reshape((path_count, paths.num_steps, output_size))
        increments = paths.wiener_increments.reshape(
            (path_count, paths.num_steps, noise_size)
        )
        dt = jnp.diff(paths.times)
        interval_controls = jnp.einsum(
            "pto,ptn->pton",
            target_flat,
            increments,
        ) / dt.reshape((1, paths.num_steps, 1, 1))
        interval_controls = interval_controls.reshape(
            paths.sample_shape
            + (paths.num_steps,)
            + problem.output_shape
            + problem.noise_shape
        )
        terminal_control = jnp.zeros(
            paths.sample_shape + (1,) + problem.output_shape + problem.noise_shape,
            dtype=interval_controls.dtype,
        )
        control_targets = jnp.concatenate(
            (interval_controls, terminal_control), axis=time_axis
        ).reshape((-1,) + problem.output_shape + problem.noise_shape)
        interval_valid = jnp.take(continuation_valid, jnp.arange(1, nodes), axis=time_axis)
        terminal_invalid = jnp.zeros(paths.sample_shape + (1,), dtype=bool)
        control_valid = jnp.concatenate(
            (interval_valid, terminal_invalid), axis=time_axis
        ).reshape((-1,))
        control_errors = jnp.full(control_targets.shape, jnp.nan, dtype=float)
    elif plan.control_target_mode == "malliavin":
        raise ValueError(
            "Trajectory-node Malliavin targets require query-conditioned continuations."
        )
    return FeynmanKacLabelBatch(
        query_times,
        query_states,
        value_targets,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        plan_id=plan.plan_id,
        value_standard_errors=jnp.full(value_targets.shape, jnp.nan, dtype=float),
        control_targets=control_targets,
        control_standard_errors=control_errors,
        valid=valid,
        control_valid=control_valid,
        sample_weights=sample_weights,
        cluster_ids=cluster_ids,
        source_path_count=1,
        metadata={
            "sampling_mode": "trajectory_nodes",
            "path_id": paths.path_id,
            "quadrature": plan.quadrature,
        },
    )


def _resolve_queries(
    problem: BSDEProblem,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    key: Key[Array, ""],
    query_times: ArrayLike | None,
    query_states: ArrayLike | None,
    query_weights: ArrayLike | None,
    query_sampler: Callable[[Key[Array, ""]], Any] | None,
) -> tuple[Array, Array, Array]:
    if query_sampler is not None:
        if query_times is not None or query_states is not None or query_weights is not None:
            raise ValueError("Supply either query_sampler or explicit queries, not both.")
        sampled = query_sampler(key)
        if isinstance(sampled, Mapping):
            if "times" not in sampled or "states" not in sampled:
                raise ValueError("Query mappings require 'times' and 'states'.")
            query_times = sampled["times"]
            query_states = sampled["states"]
            query_weights = sampled.get("weights")
        elif isinstance(sampled, tuple) and len(sampled) in (2, 3):
            query_times = sampled[0]
            query_states = sampled[1]
            query_weights = sampled[2] if len(sampled) == 3 else None
        else:
            raise TypeError("query_sampler must return a mapping or a 2/3-tuple.")
    if query_times is None or query_states is None:
        raise ValueError("Query times and states are required.")
    states = jnp.asarray(query_states)
    if states.ndim < len(problem.state_shape) + 1:
        raise ValueError("query_states require at least one query axis.")
    if states.shape[-len(problem.state_shape) :] != problem.state_shape:
        raise ValueError("query_states trailing dimensions must equal state_shape.")
    query_shape = states.shape[: -len(problem.state_shape)]
    times = jnp.asarray(query_times, dtype=float)
    if times.shape != query_shape:
        raise ValueError("query_times must match the query-state leading shape.")
    times = times.reshape((-1,))
    states = states.reshape((-1,) + problem.state_shape)
    if bool(jnp.any(~jnp.isfinite(times))) or bool(
        jnp.any((times < plan.initial_time) | (times > plan.terminal_time))
    ):
        raise ValueError("Query times must lie inside the sampling-plan interval.")
    if query_weights is None:
        weights = jnp.ones(times.shape, dtype=float)
    else:
        weights = jnp.asarray(query_weights, dtype=float)
        if weights.shape != query_shape:
            raise ValueError("query_weights must match the query shape.")
        weights = weights.reshape((-1,))
    if bool(jnp.any(~jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
        raise ValueError("Query weights must be finite and nonnegative.")
    return times, states, weights


def _normal_draws(
    key: Key[Array, ""],
    shape: tuple[int, ...],
    /,
    *,
    antithetic: bool,
) -> tuple[Array, Array]:
    query_count, path_count, steps, *noise_shape = shape
    if antithetic:
        half = path_count // 2
        base = jr.normal(key, (query_count, half, steps, *noise_shape))
        draws = jnp.concatenate((base, -base), axis=1)
        ids = jnp.concatenate(
            (
                jnp.arange(half, dtype=jnp.int32),
                jnp.arange(half, dtype=jnp.int32),
            )
        )
    else:
        draws = jr.normal(key, shape)
        ids = jnp.arange(path_count, dtype=jnp.int32)
    return draws, jnp.broadcast_to(ids, (query_count, path_count))


def sample_feynman_kac_paths(
    problem: BSDEProblem,
    query_times: ArrayLike,
    query_states: ArrayLike,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    key: Key[Array, ""] = jr.key(0),
    query_weights: ArrayLike | None = None,
    num_paths: int | None = None,
) -> FeynmanKacPathBatch:
    """Simulate Euler--Maruyama continuations on one normalized grid per query."""
    if not isinstance(problem, BSDEProblem) or not isinstance(plan, FeynmanKacSamplingPlan):
        raise TypeError("problem and plan must be BSDEProblem and FeynmanKacSamplingPlan.")
    if plan.sampling_mode != "queries":
        raise ValueError("sample_feynman_kac_paths requires queries mode.")
    q_times, q_states, q_weights = _resolve_queries(
        problem,
        plan,
        key=jr.fold_in(key, 0),
        query_times=query_times,
        query_states=query_states,
        query_weights=query_weights,
        query_sampler=None,
    )
    path_count = plan.num_paths_per_query if num_paths is None else int(num_paths)
    if path_count < 1:
        raise ValueError("num_paths must be positive.")
    if plan.antithetic and path_count % 2:
        raise ValueError("Antithetic path batches require an even path count.")
    query_count = int(q_times.shape[0])
    steps = plan.num_time_steps
    normalized = jnp.linspace(0.0, 1.0, steps + 1)
    durations = plan.terminal_time - q_times
    times = q_times[:, None] + durations[:, None] * normalized[None, :]
    dt = jnp.diff(times, axis=-1)
    draws, dependence_ids = _normal_draws(
        jr.fold_in(key, 1),
        (query_count, path_count, steps) + problem.noise_shape,
        antithetic=plan.antithetic,
    )
    sqrt_dt_shape = (query_count, 1, steps) + (1,) * len(problem.noise_shape)
    increments = draws * jnp.sqrt(dt).reshape(sqrt_dt_shape)
    initial = jnp.broadcast_to(
        q_states[:, None],
        (query_count, path_count) + problem.state_shape,
    )
    state_size = prod(problem.state_shape)
    noise_size = prod(problem.noise_shape)

    def step(current, inputs):
        time, step_dt, noise_increment = inputs
        point_times = jnp.broadcast_to(time[:, None], (query_count, path_count)).reshape((-1,))
        flat_states = current.reshape((-1,) + problem.state_shape)

        def coefficients(point_time, point_state):
            drift = jnp.asarray(problem.drift(point_time, point_state, problem.args))
            diffusion = jnp.asarray(problem.diffusion(point_time, point_state, problem.args))
            if drift.shape != problem.state_shape:
                raise ValueError("BSDE drift returned an incompatible state shape.")
            if diffusion.shape != problem.state_shape + problem.noise_shape:
                raise ValueError("BSDE diffusion returned an incompatible factor shape.")
            return drift, diffusion

        drift, diffusion = jax.vmap(coefficients)(point_times, flat_states)
        flat_increment = noise_increment.reshape((-1, noise_size))
        diffusion_action = jnp.einsum(
            "psn,pn->ps",
            diffusion.reshape((-1, state_size, noise_size)),
            flat_increment,
        ).reshape((query_count, path_count) + problem.state_shape)
        drift_increment = drift.reshape(
            (query_count, path_count) + problem.state_shape
        ) * step_dt.reshape((query_count, 1) + (1,) * len(problem.state_shape))
        next_state = current + drift_increment + diffusion_action
        return next_state, next_state

    scan_inputs = (
        jnp.moveaxis(times[:, :-1], 1, 0),
        jnp.moveaxis(dt, 1, 0),
        jnp.moveaxis(increments, 2, 0),
    )
    _, scanned = jax.lax.scan(step, initial, scan_inputs)
    states = jnp.concatenate(
        (initial[:, :, None], jnp.moveaxis(scanned, 0, 2)),
        axis=2,
    )
    valid = _event_finite(states, problem.state_shape)
    return FeynmanKacPathBatch(
        q_times,
        q_states,
        times,
        states,
        increments,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        path_id=f"feynman-kac:{problem.process_id}:{plan.plan_id}",
        process_id=problem.process_id,
        valid=valid,
        query_weights=q_weights,
        dependence_ids=dependence_ids,
        antithetic=plan.antithetic,
    )


def _aggregate_samples(
    samples: Array,
    valid: Array,
    /,
    *,
    antithetic: bool,
) -> tuple[Array, Array, Array, int]:
    path_count = int(samples.shape[1])
    if antithetic:
        half = path_count // 2
        valid_clusters = valid[:, :half] & valid[:, half:]
        cluster_samples = 0.5 * (samples[:, :half] + samples[:, half:])
        cluster_count = half
    else:
        valid_clusters = valid
        cluster_samples = samples
        cluster_count = path_count
    mask = valid_clusters.reshape(valid_clusters.shape + (1,) * (samples.ndim - 2))
    count = jnp.sum(valid_clusters, axis=1)
    safe_count = jnp.maximum(count, 1)
    mean = jnp.sum(jnp.where(mask, cluster_samples, 0.0), axis=1) / safe_count.reshape(
        safe_count.shape + (1,) * (samples.ndim - 2)
    )
    centered = jnp.where(mask, cluster_samples - mean[:, None], 0.0)
    square = jnp.abs(centered) ** 2
    variance = jnp.sum(square, axis=1) / jnp.maximum(count - 1, 1).reshape(
        count.shape + (1,) * (samples.ndim - 2)
    )
    standard_error = jnp.sqrt(
        variance
        / safe_count.reshape(safe_count.shape + (1,) * (samples.ndim - 2))
    )
    standard_error = jnp.where(
        (count >= 2).reshape(count.shape + (1,) * (samples.ndim - 2)),
        standard_error,
        jnp.nan,
    )
    return mean, standard_error, count > 0, cluster_count


def _query_path_targets(
    problem: BSDEProblem,
    paths: FeynmanKacPathBatch,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    source_value: Predictor | None,
    source_control: Predictor | None,
    key: Key[Array, ""],
) -> tuple[Array, Array, Array]:
    generator = _source_generator_nodes(
        problem,
        paths.times[:, None, :],
        paths.states,
        source_value=source_value,
        source_control=source_control,
        key=jr.fold_in(key, 0),
    )
    terminal = _terminal_values(problem, paths.states[:, :, -1])
    targets = _reverse_targets(
        terminal,
        generator,
        paths.times,
        time_axis=2,
        quadrature=plan.quadrature,
    )
    valid = _continuation_valid(paths.valid, time_axis=2)
    return targets, valid, paths.wiener_increments


def query_feynman_kac_labels(
    problem: BSDEProblem,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    query_times: ArrayLike | None = None,
    query_states: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
    query_sampler: Callable[[Key[Array, ""]], Any] | None = None,
    source_value: Predictor | None = None,
    source_control: Predictor | None = None,
    malliavin_weight: MalliavinWeight | None = None,
    key: Key[Array, ""] = jr.key(0),
    return_paths: bool = False,
) -> FeynmanKacLabelBatch | tuple[FeynmanKacLabelBatch, FeynmanKacPathBatch]:
    """Estimate conditional Feynman--Kac value/control targets at explicit queries."""
    if not isinstance(problem, BSDEProblem) or not isinstance(plan, FeynmanKacSamplingPlan):
        raise TypeError("problem and plan must be BSDEProblem and FeynmanKacSamplingPlan.")
    if plan.sampling_mode != "queries":
        raise ValueError("query_feynman_kac_labels requires queries mode.")
    q_times, q_states, q_weights = _resolve_queries(
        problem,
        plan,
        key=jr.fold_in(key, 0),
        query_times=query_times,
        query_states=query_states,
        query_weights=query_weights,
        query_sampler=query_sampler,
    )
    paths = sample_feynman_kac_paths(
        problem,
        q_times,
        q_states,
        plan,
        key=jr.fold_in(key, 1),
        query_weights=q_weights,
    )
    targets, continuation_valid, increments = _query_path_targets(
        problem,
        paths,
        plan,
        source_value=source_value,
        source_control=source_control,
        key=jr.fold_in(key, 2),
    )
    value_samples = targets[:, :, 0]
    path_valid = continuation_valid[:, :, 0] & _event_finite(
        value_samples, problem.output_shape
    )
    value_targets, value_errors, valid, cluster_count = _aggregate_samples(
        value_samples,
        path_valid,
        antithetic=plan.antithetic,
    )
    control_targets = None
    control_errors = None
    control_valid = jnp.zeros_like(valid)
    duration = plan.terminal_time - q_times
    nonterminal = duration > 0.0
    if plan.control_target_mode == "martingale":
        output_size = prod(problem.output_shape)
        noise_size = prod(problem.noise_shape)
        next_values = targets[:, :, 1].reshape(
            (q_times.shape[0], paths.num_paths, output_size)
        )
        first_increment = increments[:, :, 0].reshape(
            (q_times.shape[0], paths.num_paths, noise_size)
        )
        dt0 = paths.times[:, 1] - paths.times[:, 0]
        safe_dt = jnp.where(nonterminal, dt0, 1.0)
        samples = jnp.einsum(
            "qpo,qpn->qpon",
            next_values,
            first_increment,
        ) / safe_dt.reshape((-1, 1, 1, 1))
        samples = samples.reshape(
            (q_times.shape[0], paths.num_paths)
            + problem.output_shape
            + problem.noise_shape
        )
        control_path_valid = continuation_valid[:, :, 1] & nonterminal[:, None]
        control_targets, control_errors, control_valid, _ = _aggregate_samples(
            samples,
            control_path_valid,
            antithetic=plan.antithetic,
        )
        control_targets = jnp.where(
            nonterminal.reshape((-1,) + (1,) * (len(problem.output_shape) + len(problem.noise_shape))),
            control_targets,
            0.0,
        )
    elif plan.control_target_mode == "malliavin":
        if malliavin_weight is None:
            raise ValueError("Malliavin control targets require malliavin_weight.")
        total_increment = jnp.sum(increments, axis=2)
        flat_weights = jax.vmap(
            jax.vmap(
                lambda state, path_state, increment, time: jnp.asarray(
                    malliavin_weight(time, state, path_state, increment, problem.args)
                ),
                in_axes=(None, 0, 0, None),
            ),
            in_axes=(0, 0, 0, 0),
        )(q_states, paths.states[:, :, -1], total_increment, q_times)
        expected_weights = (
            q_times.shape[0],
            paths.num_paths,
        ) + problem.noise_shape
        if flat_weights.shape != expected_weights:
            raise ValueError(
                f"malliavin_weight must return trailing shape {problem.noise_shape}."
            )
        baseline = _terminal_values(problem, q_states)
        centered = value_samples - baseline[:, None]
        samples = jnp.einsum(
            "qpo,qpn->qpon",
            centered.reshape((q_times.shape[0], paths.num_paths, -1)),
            flat_weights.reshape((q_times.shape[0], paths.num_paths, -1)),
        ).reshape(
            (q_times.shape[0], paths.num_paths)
            + problem.output_shape
            + problem.noise_shape
        )
        control_targets, control_errors, control_valid, _ = _aggregate_samples(
            samples,
            path_valid & nonterminal[:, None],
            antithetic=plan.antithetic,
        )
    labels = FeynmanKacLabelBatch(
        q_times,
        q_states,
        value_targets,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        plan_id=plan.plan_id,
        value_standard_errors=value_errors,
        control_targets=control_targets,
        control_standard_errors=control_errors,
        valid=valid,
        control_valid=control_valid,
        sample_weights=q_weights,
        cluster_ids=jnp.arange(q_times.shape[0], dtype=jnp.int32),
        source_path_count=cluster_count,
        metadata={
            "sampling_mode": "queries",
            "quadrature": plan.quadrature,
            "antithetic": plan.antithetic,
            "path_id": paths.path_id,
        },
    )
    if return_paths:
        return labels, paths
    return labels


__all__ = [
    "FeynmanKacControlTargetMode",
    "FeynmanKacLabelBatch",
    "FeynmanKacLabelDiagnostics",
    "feynman_kac_label_diagnostics",
    "FeynmanKacPathBatch",
    "FeynmanKacRefreshMode",
    "FeynmanKacSamplingMode",
    "FeynmanKacSamplingPlan",
    "FeynmanKacTimeWeighting",
    "query_feynman_kac_labels",
    "sample_feynman_kac_paths",
    "trajectory_node_feynman_kac_labels",
]
