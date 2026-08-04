#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from ._bsde import (
    BSDEControlMode,
    BSDEEvaluation,
    BSDEObjectiveMode,
    BSDEPathBatch,
    BSDEProblem,
    BSDEQuadrature,
    evaluate_bsde,
)
from ._jump import JUMP_SUCCESS, JumpEventBatch, PoissonClockRealization
from ._realization import CompositeStochasticRealization
from ._wiener import WienerRealization


class JumpBSDEProblem(StrictModule):
    """Finite-activity jump extension of one Brownian BSDE problem."""

    base: BSDEProblem
    compensator_rate: Callable[[str, Array, Array, Callable, Any], Array]
    jump_process_ids: frozendict[str, str] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: BSDEProblem,
        compensator_rate: Callable[[str, Array, Array, Callable, Any], Array],
        jump_process_ids: Mapping[str, str],
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(base, BSDEProblem):
            raise TypeError("base must be a BSDEProblem.")
        if not callable(compensator_rate):
            raise TypeError("compensator_rate must be callable.")
        process_ids = frozendict(jump_process_ids)
        if not process_ids:
            raise ValueError("jump_process_ids must not be empty.")
        for label, process_id in process_ids.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Jump labels must be non-empty strings.")
            if not isinstance(process_id, str) or not process_id:
                raise ValueError("Jump process IDs must be non-empty strings.")
        resolved_id = base.problem_id if problem_id is None else problem_id
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("problem_id must be a non-empty string.")
        self.base = base
        self.compensator_rate = compensator_rate
        self.jump_process_ids = process_ids
        self.problem_id = resolved_id


class JumpBSDEEvaluation(StrictModule):
    """Brownian BSDE evaluation augmented by compensated jump increments."""

    base: BSDEEvaluation
    jump_sums: Array
    compensator_increments: Array
    compensated_jump_increments: Array
    local_residuals: Array
    global_residual: Array
    valid_paths: Array
    event_counts: frozendict[str, Array]
    event_status: frozendict[str, Array]
    problem_id: str = eqx.field(static=True)


class JumpBSDEDiagnostics(StrictModule):
    """Finite-activity event and compensated-residual diagnostics."""

    mean_event_count: frozendict[str, Array]
    failure_fraction: frozendict[str, Array]
    compensated_increment_mean: Array
    local_residual_rms: Array
    global_residual_rms: Array
    num_valid: int = eqx.field(static=True)
    passed: bool = eqx.field(static=True)


def _validate_jump_provenance(
    problem: JumpBSDEProblem,
    paths: BSDEPathBatch,
    /,
) -> None:
    if set(paths.jump_events) != set(problem.jump_process_ids):
        raise ValueError("BSDE jump-event labels must exactly match jump_process_ids.")
    for label, events in paths.jump_events.items():
        if not isinstance(events, JumpEventBatch):
            raise TypeError(f"jump_events[{label!r}] must be a JumpEventBatch.")
        if events.batch_shape != paths.sample_shape:
            raise ValueError(f"jump_events[{label!r}] has incompatible batch_shape.")
        if events.state_shape != paths.state_shape:
            raise ValueError(f"jump_events[{label!r}] has incompatible state_shape.")
        if events.pre_states is None:
            raise ValueError(
                f"jump_events[{label!r}] must retain pre_states for jump BSDEs."
            )
    realization = paths.realization
    if realization is None:
        return
    if not isinstance(realization, CompositeStochasticRealization):
        raise ValueError(
            "Paths with jump events require a CompositeStochasticRealization "
            "when realization provenance is present."
        )
    wiener_components = tuple(
        component
        for component in realization.components.values()
        if isinstance(component, WienerRealization)
        and component.noise_id == paths.process_id
    )
    if len(wiener_components) != 1:
        raise ValueError(
            "Composite BSDE provenance requires exactly one matching Wiener component."
        )
    for label, process_id in problem.jump_process_ids.items():
        if label not in realization.components:
            raise ValueError(
                f"Composite realization is missing jump component {label!r}."
            )
        component = realization.component(label)
        if not isinstance(component, PoissonClockRealization):
            raise TypeError(f"Realization component {label!r} must be a Poisson clock.")
        if component.process_id != process_id:
            raise ValueError(f"Jump process ID mismatch for component {label!r}.")
        if component.sample_shape != paths.sample_shape:
            raise ValueError(f"Jump realization {label!r} has incompatible sample_shape.")


def _jump_values(
    jump_control: Callable,
    label: str,
    events: JumpEventBatch,
    problem: JumpBSDEProblem,
    /,
    *,
    key: Key[Array, ""],
) -> Array:
    pre_states = events.pre_states
    if pre_states is None:
        raise ValueError("Jump BSDE events must retain pre_states.")
    sample_size = int(jnp.prod(jnp.asarray(events.batch_shape)))
    flat_times = events.times.reshape((sample_size, events.max_events))
    flat_states = pre_states.reshape(
        (sample_size, events.max_events) + events.state_shape
    )
    flat_channels = events.channels.reshape((sample_size, events.max_events))
    flat_marks = events.marks.reshape(
        (sample_size, events.max_events) + events.mark_shape
    )
    path_indices = jnp.arange(sample_size, dtype=jnp.uint32)
    event_indices = jnp.arange(events.max_events, dtype=jnp.uint32)

    def one_path(path_index, times, states, channels, marks):
        def one_event(event_index, time, state, channel, mark):
            event_key = jax.random.fold_in(
                key, path_index * events.max_events + event_index
            )
            value = jnp.asarray(
                jump_control(
                    label,
                    time,
                    state,
                    channel,
                    mark,
                    problem.base.args,
                    key=event_key,
                )
            )
            return value

        return jax.vmap(one_event)(event_indices, times, states, channels, marks)

    values = jax.vmap(one_path)(
        path_indices,
        flat_times,
        flat_states,
        flat_channels,
        flat_marks,
    )
    expected = (sample_size, events.max_events) + problem.base.output_shape
    if values.shape != expected:
        raise ValueError(
            f"jump_control returned shape {values.shape}; expected {expected}."
        )
    return values.reshape(
        events.batch_shape + (events.max_events,) + problem.base.output_shape
    )


def _compensator_rates(
    problem: JumpBSDEProblem,
    paths: BSDEPathBatch,
    jump_control: Callable,
    label: str,
    /,
    *,
    quadrature: BSDEQuadrature,
) -> Array:
    times = paths.times
    states = paths.states
    state_prefix = (slice(None),) * len(paths.sample_shape)
    state_suffix = (slice(None),) * len(paths.state_shape)
    left_states = states[state_prefix + (slice(None, -1),) + state_suffix]
    right_states = states[state_prefix + (slice(1, None),) + state_suffix]
    if quadrature == "left":
        eval_times = times[:-1]
        eval_states = left_states
    elif quadrature == "midpoint":
        eval_times = 0.5 * (times[:-1] + times[1:])
        eval_states = 0.5 * (left_states + right_states)
    elif quadrature == "trapezoid":
        left = _compensator_rates(problem, paths, jump_control, label, quadrature="left")
        node_rates = _node_compensator_rates(problem, paths, jump_control, label)
        rate_prefix = (slice(None),) * len(paths.sample_shape)
        rate_suffix = (slice(None),) * len(problem.base.output_shape)
        right_rates = node_rates[rate_prefix + (slice(1, None),) + rate_suffix]
        return 0.5 * (left + right_rates)
    else:
        raise ValueError("quadrature must be 'left', 'midpoint', or 'trapezoid'.")
    sample_size = paths.num_paths
    flat_states = eval_states.reshape((sample_size, paths.num_steps) + paths.state_shape)

    def one_path(path_states):
        return jax.vmap(
            lambda time, state: jnp.asarray(
                problem.compensator_rate(
                    label, time, state, jump_control, problem.base.args
                )
            )
        )(eval_times, path_states)

    rates = jax.vmap(one_path)(flat_states)
    expected = (sample_size, paths.num_steps) + problem.base.output_shape
    if rates.shape != expected:
        raise ValueError(
            f"compensator_rate returned shape {rates.shape}; expected {expected}."
        )
    return rates.reshape(
        paths.sample_shape + (paths.num_steps,) + problem.base.output_shape
    )


def _node_compensator_rates(
    problem: JumpBSDEProblem,
    paths: BSDEPathBatch,
    jump_control: Callable,
    label: str,
    /,
) -> Array:
    sample_size = paths.num_paths
    flat_states = paths.states.reshape(
        (sample_size, paths.times.shape[0]) + paths.state_shape
    )

    def one_path(path_states):
        return jax.vmap(
            lambda time, state: jnp.asarray(
                problem.compensator_rate(
                    label, time, state, jump_control, problem.base.args
                )
            )
        )(paths.times, path_states)

    rates = jax.vmap(one_path)(flat_states)
    expected = (sample_size, paths.times.shape[0]) + problem.base.output_shape
    if rates.shape != expected:
        raise ValueError(
            f"compensator_rate returned shape {rates.shape}; expected {expected}."
        )
    return rates.reshape(
        paths.sample_shape + (paths.times.shape[0],) + problem.base.output_shape
    )


def evaluate_jump_bsde(
    problem: JumpBSDEProblem,
    paths: BSDEPathBatch,
    value_predictor: Callable,
    jump_control: Callable,
    /,
    *,
    control_predictor: Callable | None = None,
    control_mode: BSDEControlMode = "explicit",
    quadrature: BSDEQuadrature = "left",
    key: Key[Array, ""] = jax.random.key(0),
    raise_on_failure: bool = False,
) -> JumpBSDEEvaluation:
    """Evaluate a finite-activity BSDE using compensated jump increments."""
    if not isinstance(problem, JumpBSDEProblem):
        raise TypeError("problem must be a JumpBSDEProblem.")
    if not isinstance(paths, BSDEPathBatch):
        raise TypeError("paths must be a BSDEPathBatch.")
    if not callable(jump_control):
        raise TypeError("jump_control must be callable.")
    _validate_jump_provenance(problem, paths)
    base = evaluate_bsde(
        problem.base,
        paths,
        value_predictor,
        control_predictor=control_predictor,
        control_mode=control_mode,
        quadrature=quadrature,
        key=key,
    )
    jump_sums = jnp.zeros_like(base.local_residuals)
    compensators = jnp.zeros_like(base.local_residuals)
    event_counts: dict[str, Array] = {}
    event_status: dict[str, Array] = {}
    successful = jnp.ones(paths.sample_shape, dtype=bool)
    times = paths.times
    output_axes = (1,) * len(problem.base.output_shape)
    for label, events in paths.jump_events.items():
        values = _jump_values(jump_control, label, events, problem, key=key)
        event_mask = events.valid
        interval_mask = (
            event_mask[..., None, :]
            & (events.times[..., None, :] > times[:-1, None])
            & (events.times[..., None, :] <= times[1:, None])
        )
        value_expanded = values[..., None, :, *([slice(None)] * len(output_axes))]
        mask_expanded = interval_mask.reshape(interval_mask.shape + output_axes)
        jump_sums = jump_sums + jnp.sum(
            jnp.where(mask_expanded, value_expanded, 0.0), axis=-1 - len(output_axes)
        )
        rates = _compensator_rates(
            problem,
            paths,
            jump_control,
            label,
            quadrature=quadrature,
        )
        dt_shape = (1,) * len(paths.sample_shape) + (paths.num_steps,) + output_axes
        compensators = compensators + rates * jnp.diff(times).reshape(dt_shape)
        event_counts[label] = events.counts
        event_status[label] = events.status
        successful = successful & (events.status == JUMP_SUCCESS)
    compensated = jump_sums - compensators
    local = base.local_residuals - compensated
    global_residual = base.global_residual - jnp.sum(compensated, axis=-2)
    valid_paths = base.valid_paths & successful
    result = JumpBSDEEvaluation(
        base=base,
        jump_sums=jump_sums,
        compensator_increments=compensators,
        compensated_jump_increments=compensated,
        local_residuals=local,
        global_residual=global_residual,
        valid_paths=valid_paths,
        event_counts=frozendict(event_counts),
        event_status=frozendict(event_status),
        problem_id=problem.problem_id,
    )
    if raise_on_failure and not bool(jnp.all(valid_paths)):
        raise RuntimeError("Jump BSDE evaluation contains failed event paths.")
    return result


def jump_bsde_objective_loss(
    evaluation: JumpBSDEEvaluation,
    /,
    *,
    mode: BSDEObjectiveMode = "joint",
    terminal_weight: float = 1.0,
    local_weight: float = 1.0,
    global_weight: float = 1.0,
) -> Array:
    """Masked mean-square objective for a compensated jump BSDE evaluation."""
    if not isinstance(evaluation, JumpBSDEEvaluation):
        raise TypeError("evaluation must be a JumpBSDEEvaluation.")
    if mode not in ("terminal", "local", "global", "joint"):
        raise ValueError("mode must be 'terminal', 'local', 'global', or 'joint'.")
    weights = (float(terminal_weight), float(local_weight), float(global_weight))
    if any(not jnp.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("BSDE objective weights must be finite and nonnegative.")
    valid = evaluation.valid_paths
    count = jnp.maximum(jnp.sum(valid), 1)

    def masked_square(values: Array) -> Array:
        expanded = valid.reshape(valid.shape + (1,) * (values.ndim - valid.ndim))
        event_size = int(jnp.prod(jnp.asarray(values.shape[valid.ndim :])))
        return jnp.sum(jnp.where(expanded, values**2, 0.0)) / (count * event_size)

    terminal_loss = masked_square(evaluation.base.terminal_residual)
    local_loss = masked_square(evaluation.local_residuals)
    global_loss = masked_square(evaluation.global_residual)
    if mode == "terminal":
        return weights[0] * terminal_loss
    if mode == "local":
        return weights[1] * local_loss
    if mode == "global":
        return weights[2] * global_loss
    return weights[0] * terminal_loss + weights[1] * local_loss + weights[2] * global_loss


def jump_bsde_diagnostics(
    evaluation: JumpBSDEEvaluation,
    /,
) -> JumpBSDEDiagnostics:
    """Summarize event failures and compensated BSDE residual scale."""
    if not isinstance(evaluation, JumpBSDEEvaluation):
        raise TypeError("evaluation must be a JumpBSDEEvaluation.")
    valid = evaluation.valid_paths
    num_valid = int(jnp.sum(valid))
    mean_counts = {
        label: jnp.mean(counts.astype(float))
        for label, counts in evaluation.event_counts.items()
    }
    failures = {
        label: jnp.mean((status != JUMP_SUCCESS).astype(float))
        for label, status in evaluation.event_status.items()
    }
    compensated_mean = jnp.mean(
        evaluation.compensated_jump_increments,
        axis=tuple(range(evaluation.compensated_jump_increments.ndim - 1)),
    )
    local_rms = jnp.sqrt(jnp.mean(evaluation.local_residuals**2))
    global_rms = jnp.sqrt(jnp.mean(evaluation.global_residual**2))
    passed = bool(
        num_valid > 0
        and jnp.all(jnp.isfinite(evaluation.local_residuals))
        and jnp.all(jnp.isfinite(evaluation.global_residual))
        and all(float(fraction) == 0.0 for fraction in failures.values())
    )
    return JumpBSDEDiagnostics(
        mean_event_count=frozendict(mean_counts),
        failure_fraction=frozendict(failures),
        compensated_increment_mean=compensated_mean,
        local_residual_rms=local_rms,
        global_residual_rms=global_rms,
        num_valid=num_valid,
        passed=passed,
    )


__all__ = [
    "evaluate_jump_bsde",
    "jump_bsde_diagnostics",
    "jump_bsde_objective_loss",
    "JumpBSDEDiagnostics",
    "JumpBSDEEvaluation",
    "JumpBSDEProblem",
]
