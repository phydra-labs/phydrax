#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._objective import AbstractSamplingObjectiveTerm
from .._strict import StrictModule
from ..stochastic._bsde import _pointwise_values, BSDEPathBatch, BSDEProblem


DeepBSDESamplingMode: TypeAlias = Literal["resample", "fixed"]
DeepBSDEPredictor: TypeAlias = Callable | DomainFunction


def _validate_paths(paths: BSDEPathBatch, problem: BSDEProblem, /) -> None:
    if not isinstance(paths, BSDEPathBatch):
        raise TypeError("Deep BSDE paths must be a BSDEPathBatch.")
    if paths.state_shape != problem.state_shape or paths.noise_shape != problem.noise_shape:
        raise ValueError("Deep BSDE path event shapes do not match the problem.")
    if paths.process_id != problem.process_id:
        raise ValueError("Deep BSDE path and problem process IDs do not match.")


def _path_mask(paths: BSDEPathBatch, event_ndim: int, /) -> Array:
    return paths.path_valid.reshape(paths.sample_shape + (1,) * (event_ndim + 1))


def _event_finite(values: Array, event_shape: tuple[int, ...], /) -> Array:
    axes = tuple(range(values.ndim - len(event_shape), values.ndim))
    return jnp.all(jnp.isfinite(values), axis=axes)


def _masked_event_mean(
    values: Array,
    valid: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    sample_axes = tuple(range(values.ndim - len(event_shape)))
    mask = valid.reshape(valid.shape + (1,) * len(event_shape))
    count = jnp.maximum(jnp.sum(valid), 1)
    return jnp.sum(jnp.where(mask, values, 0.0), axis=sample_axes) / count


def _masked_mean_square(
    values: Array,
    valid: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    squared = jnp.abs(values) ** 2
    event_axes = tuple(range(squared.ndim - len(event_shape), squared.ndim))
    squared = jnp.sum(squared, axis=event_axes)
    count = jnp.sum(valid)
    count = eqx.error_if(
        count,
        ~(jnp.isfinite(count) & (count > 0)),
        "Deep BSDE batch has no valid paths.",
    )
    return jnp.sum(jnp.where(valid, squared, 0.0)) / count


class DeepBSDERollout(StrictModule):
    """Learned forward shooting trajectory and terminal mismatch."""

    initial_values: Array
    values: Array
    controls: Array
    generator_values: Array
    martingale_increments: Array
    terminal_targets: Array
    terminal_residual: Array
    valid_paths: Array
    paths: BSDEPathBatch


class DeepBSDEShootingDiagnostics(StrictModule):
    """Independent-path diagnostics for one learned shooting solution."""

    terminal_rmse: Array
    terminal_bias: Array
    initial_mean: Array
    control_rms: Array
    valid_fraction: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


def deep_bsde_rollout(
    problem: BSDEProblem,
    paths: BSDEPathBatch,
    initial_value_predictor: DeepBSDEPredictor,
    control_predictor: DeepBSDEPredictor,
    /,
    *,
    key: Key[Array, ""] = jr.key(0),
) -> DeepBSDERollout:
    """Roll Y forward with Y[n+1] = Y[n] - f[n] dt + Z[n] dW[n]."""
    if not isinstance(problem, BSDEProblem):
        raise TypeError("problem must be a BSDEProblem.")
    _validate_paths(paths, problem)
    if not callable(initial_value_predictor) or not callable(control_predictor):
        raise TypeError("Deep BSDE initial-value and control predictors must be callable.")

    sample_count = prod(paths.sample_shape) if paths.sample_shape else 1
    state_size = prod(problem.state_shape)
    output_size = prod(problem.output_shape)
    noise_size = prod(problem.noise_shape)
    state_mask = _path_mask(paths, len(problem.state_shape))
    increment_mask = _path_mask(paths, len(problem.noise_shape))
    safe_states = jnp.where(state_mask, paths.states, 0.0)
    safe_increments = jnp.where(increment_mask, paths.wiener_increments, 0.0)
    initial_key, control_key = jr.split(key)
    initial_times = jnp.full(paths.sample_shape, paths.times[0])
    initial_states = safe_states[
        ..., 0, *([slice(None)] * len(problem.state_shape))
    ]
    initial_values = _pointwise_values(
        initial_value_predictor,
        initial_times,
        initial_states,
        problem,
        key=initial_key,
        output_shape=problem.output_shape,
    )
    left_states = safe_states[
        ..., :-1, *([slice(None)] * len(problem.state_shape))
    ]
    controls = _pointwise_values(
        control_predictor,
        paths.times[:-1],
        left_states,
        problem,
        key=control_key,
        output_shape=problem.output_shape + problem.noise_shape,
    )

    initial_flat = initial_values.reshape((sample_count, output_size))
    states_flat = left_states.reshape(
        (sample_count, paths.num_steps, state_size)
    )
    controls_flat = controls.reshape(
        (sample_count, paths.num_steps, output_size, noise_size)
    )
    increments_flat = safe_increments.reshape(
        (sample_count, paths.num_steps, noise_size)
    )

    def step(value_flat, inputs):
        time, state_flat, control_flat, increment_flat, dt = inputs

        def generator_at(state, value, control):
            output = jnp.asarray(
                problem.generator(
                    time,
                    state.reshape(problem.state_shape),
                    value.reshape(problem.output_shape),
                    control.reshape(problem.output_shape + problem.noise_shape),
                    problem.args,
                )
            )
            if output.shape != problem.output_shape:
                raise ValueError("BSDE generator returned an incompatible output shape.")
            return output.reshape((output_size,))

        generator_flat = jax.vmap(generator_at)(state_flat, value_flat, control_flat)
        martingale_flat = jnp.einsum(
            "son,sn->so", control_flat, increment_flat
        )
        next_value = value_flat - dt * generator_flat + martingale_flat
        return next_value, (next_value, generator_flat, martingale_flat)

    _, (next_values, generator_values, martingale_increments) = jax.lax.scan(
        step,
        initial_flat,
        (
            paths.times[:-1],
            jnp.moveaxis(states_flat, 1, 0),
            jnp.moveaxis(controls_flat, 1, 0),
            jnp.moveaxis(increments_flat, 1, 0),
            jnp.diff(paths.times),
        ),
    )
    next_values = jnp.moveaxis(next_values, 0, 1)
    values_flat = jnp.concatenate((initial_flat[:, None, :], next_values), axis=1)
    values = values_flat.reshape(
        paths.sample_shape + (paths.num_steps + 1,) + problem.output_shape
    )
    generator_values = jnp.moveaxis(generator_values, 0, 1).reshape(
        paths.sample_shape + (paths.num_steps,) + problem.output_shape
    )
    martingale_increments = jnp.moveaxis(martingale_increments, 0, 1).reshape(
        paths.sample_shape + (paths.num_steps,) + problem.output_shape
    )
    terminal_states = safe_states[
        ..., -1, *([slice(None)] * len(problem.state_shape))
    ]
    terminal_targets = jax.vmap(
        lambda state: jnp.asarray(problem.terminal(state, problem.args))
    )(terminal_states.reshape((-1,) + problem.state_shape)).reshape(
        paths.sample_shape + problem.output_shape
    )
    if terminal_targets.shape != paths.sample_shape + problem.output_shape:
        raise ValueError("BSDE terminal condition returned an incompatible output shape.")
    terminal_values = values[
        ..., -1, *([slice(None)] * len(problem.output_shape))
    ]
    terminal_residual = terminal_values - terminal_targets
    valid_paths = (
        paths.path_valid
        & jnp.all(_event_finite(values, problem.output_shape), axis=-1)
        & jnp.all(
            _event_finite(controls, problem.output_shape + problem.noise_shape),
            axis=-1,
        )
        & jnp.all(_event_finite(generator_values, problem.output_shape), axis=-1)
        & _event_finite(terminal_residual, problem.output_shape)
    )
    return DeepBSDERollout(
        initial_values=initial_values,
        values=values,
        controls=controls,
        generator_values=generator_values,
        martingale_increments=martingale_increments,
        terminal_targets=terminal_targets,
        terminal_residual=terminal_residual,
        valid_paths=valid_paths,
        paths=paths,
    )


def deep_bsde_shooting_diagnostics(
    rollout: DeepBSDERollout,
    /,
) -> DeepBSDEShootingDiagnostics:
    """Summarize terminal accuracy without reusing training paths implicitly."""
    if not isinstance(rollout, DeepBSDERollout):
        raise TypeError("rollout must be a DeepBSDERollout.")
    output_shape = rollout.terminal_residual.shape[len(rollout.paths.sample_shape) :]
    terminal_mse = _masked_mean_square(
        rollout.terminal_residual,
        rollout.valid_paths,
        output_shape,
    )
    terminal_bias = _masked_event_mean(
        rollout.terminal_residual,
        rollout.valid_paths,
        output_shape,
    )
    initial_mean = _masked_event_mean(
        rollout.initial_values,
        rollout.valid_paths,
        output_shape,
    )
    control_event_shape = rollout.controls.shape[
        len(rollout.paths.sample_shape) + 1 :
    ]
    interval_valid = jnp.broadcast_to(
        rollout.valid_paths[..., None],
        rollout.paths.sample_shape + (rollout.paths.num_steps,),
    )
    control_rms = jnp.sqrt(
        _masked_mean_square(rollout.controls, interval_valid, control_event_shape)
    )
    finite = jnp.asarray(
        jnp.isfinite(terminal_mse)
        & jnp.all(jnp.isfinite(terminal_bias))
        & jnp.all(jnp.isfinite(initial_mean))
        & jnp.isfinite(control_rms)
    )
    return DeepBSDEShootingDiagnostics(
        terminal_rmse=jnp.sqrt(terminal_mse),
        terminal_bias=terminal_bias,
        initial_mean=initial_mean,
        control_rms=control_rms,
        valid_fraction=jnp.mean(rollout.valid_paths),
        finite=finite,
    )


class DeepBSDEShootingObjective(AbstractSamplingObjectiveTerm):
    """Canonical terminal-mismatch objective for learned Deep BSDE shooting."""

    problem: BSDEProblem
    fixed_paths: BSDEPathBatch | None
    terminal_weight: Array
    initial_value_name: str = eqx.field(static=True)
    control_name: str = eqx.field(static=True)
    sampling_mode: DeepBSDESamplingMode = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        problem: BSDEProblem,
        /,
        *,
        initial_value_name: str,
        control_name: str,
        terminal_weight: ArrayLike = 1.0,
        sampling_mode: DeepBSDESamplingMode = "resample",
        fixed_paths: BSDEPathBatch | None = None,
        fixed_paths_key: Key[Array, ""] = jr.key(0),
        label: str | None = None,
    ):
        if not isinstance(problem, BSDEProblem):
            raise TypeError("problem must be a BSDEProblem.")
        for owner, value in (
            ("initial_value_name", initial_value_name),
            ("control_name", control_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{owner} must be a non-empty string.")
        if initial_value_name == control_name:
            raise ValueError("Initial-value and control function names must be distinct.")
        if sampling_mode not in ("resample", "fixed"):
            raise ValueError("sampling_mode must be 'resample' or 'fixed'.")
        if fixed_paths is not None:
            _validate_paths(fixed_paths, problem)
        if sampling_mode == "resample" and fixed_paths is not None:
            raise ValueError("fixed_paths is valid only for fixed sampling.")
        weight = jnp.asarray(terminal_weight, dtype=float).reshape(())
        if bool(~jnp.isfinite(weight)) or float(weight) < 0.0:
            raise ValueError("terminal_weight must be finite and nonnegative.")
        self.problem = problem
        self.fixed_paths = (
            problem.sample(fixed_paths_key)
            if sampling_mode == "fixed" and fixed_paths is None
            else fixed_paths
        )
        self.terminal_weight = weight
        self.initial_value_name = initial_value_name
        self.control_name = control_name
        self.sampling_mode = sampling_mode
        self.label = None if label is None else str(label)

    def sample(self, *, key: Key[Array, ""] = jr.key(0)) -> BSDEPathBatch:
        if self.sampling_mode == "fixed":
            if self.fixed_paths is None:
                raise RuntimeError("Fixed Deep BSDE paths are unavailable.")
            return self.fixed_paths
        return self.problem.sample(key)

    def rollout(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: BSDEPathBatch | None = None,
    ) -> DeepBSDERollout:
        if self.initial_value_name not in functions:
            raise KeyError(
                f"Missing Deep BSDE initial-value function {self.initial_value_name!r}."
            )
        if self.control_name not in functions:
            raise KeyError(f"Missing Deep BSDE control function {self.control_name!r}.")
        sampling_key, rollout_key = jr.split(key)
        paths = self.sample(key=sampling_key) if batch is None else batch
        return deep_bsde_rollout(
            self.problem,
            paths,
            functions[self.initial_value_name],
            functions[self.control_name],
            key=rollout_key,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: BSDEPathBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        rollout = self.rollout(functions, key=key, batch=batch)
        output_shape = rollout.terminal_residual.shape[
            len(rollout.paths.sample_shape) :
        ]
        return self.terminal_weight * _masked_mean_square(
            rollout.terminal_residual,
            rollout.valid_paths,
            output_shape,
        )

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: BSDEPathBatch | None = None,
    ) -> DeepBSDEShootingDiagnostics:
        return deep_bsde_shooting_diagnostics(
            self.rollout(functions, key=key, batch=batch)
        )


__all__ = [
    "deep_bsde_rollout",
    "deep_bsde_shooting_diagnostics",
    "DeepBSDEPredictor",
    "DeepBSDERollout",
    "DeepBSDESamplingMode",
    "DeepBSDEShootingDiagnostics",
    "DeepBSDEShootingObjective",
]
