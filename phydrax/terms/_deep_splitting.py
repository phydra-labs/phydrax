#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import prod
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..stochastic._bsde import (
    _pointwise_autodiff_control,
    _pointwise_values,
    BSDEPathBatch,
    BSDEProblem,
)


DeepSplittingPredictor: TypeAlias = Callable | DomainFunction


def _event_finite(values: Array, event_shape: tuple[int, ...], /) -> Array:
    axes = tuple(range(values.ndim - len(event_shape), values.ndim))
    return jnp.all(jnp.isfinite(values), axis=axes)


def _masked_mean_square(
    values: Array,
    valid: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    event_mask = jnp.broadcast_to(
        valid.reshape(valid.shape + (1,) * len(event_shape)),
        values.shape,
    )
    safe_values = jnp.where(
        event_mask,
        values,
        jnp.zeros((), dtype=values.dtype),
    )
    squared = jnp.abs(safe_values) ** 2
    event_axes = tuple(range(squared.ndim - len(event_shape), squared.ndim))
    squared = jnp.sum(squared, axis=event_axes)
    count = jnp.sum(valid)
    count = eqx.error_if(
        count,
        ~(jnp.isfinite(count) & (count > 0)),
        "Deep splitting batch has no valid transitions.",
    )
    return jnp.sum(jnp.where(valid, squared, 0.0)) / count


def _rms(values: Array, valid: Array, event_shape: tuple[int, ...], /) -> Array:
    return jnp.sqrt(_masked_mean_square(values, valid, event_shape))


class DeepSplittingLabelBatch(StrictModule):
    """Frozen one-step Bellman targets for one backward time slice."""

    left_states: Array
    right_states: Array
    next_values: Array
    source_controls: Array
    source_values: Array
    value_targets: Array
    valid: Array
    left_time: Array
    right_time: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    slice_index: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)


class DeepSplittingRegressionDiagnostics(StrictModule):
    """One-step regression and target-scale diagnostics for a time slice."""

    one_step_rmse: Array
    target_rms: Array
    source_rms: Array
    valid_fraction: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


def _validate_paths(paths: BSDEPathBatch, problem: BSDEProblem, /) -> None:
    if not isinstance(paths, BSDEPathBatch):
        raise TypeError("Deep splitting paths must be a BSDEPathBatch.")
    if (
        paths.state_shape != problem.state_shape
        or paths.noise_shape != problem.noise_shape
    ):
        raise ValueError("Deep splitting path event shapes do not match the problem.")
    if paths.process_id != problem.process_id:
        raise ValueError("Deep splitting path and problem process IDs do not match.")


def _validate_labels(
    labels: DeepSplittingLabelBatch,
    problem: BSDEProblem,
    slice_index: int,
    /,
) -> None:
    if not isinstance(labels, DeepSplittingLabelBatch):
        raise TypeError("Deep splitting providers must return DeepSplittingLabelBatch.")
    if labels.problem_id != problem.problem_id or labels.process_id != problem.process_id:
        raise ValueError("Deep splitting label provenance does not match the problem.")
    if labels.slice_index != slice_index:
        raise ValueError("Deep splitting label slice does not match the objective.")
    if (
        labels.state_shape != problem.state_shape
        or labels.noise_shape != problem.noise_shape
        or labels.output_shape != problem.output_shape
    ):
        raise ValueError("Deep splitting label event shapes do not match the problem.")


def deep_splitting_labels(
    problem: BSDEProblem,
    paths: BSDEPathBatch,
    next_value_predictor: DeepSplittingPredictor,
    slice_index: int,
    /,
    *,
    key: Key[Array, ""] = jr.key(0),
) -> DeepSplittingLabelBatch:
    """Build the explicit right-endpoint target U[n+1] + dt f[n+1]."""
    if not isinstance(problem, BSDEProblem):
        raise TypeError("problem must be a BSDEProblem.")
    _validate_paths(paths, problem)
    if not callable(next_value_predictor):
        raise TypeError("next_value_predictor must be callable.")
    index = int(slice_index)
    if index < 0 or index >= paths.num_steps:
        raise ValueError("slice_index must identify a path interval.")
    left_time = paths.times[index]
    right_time = paths.times[index + 1]
    left_states = paths.states[..., index, *([slice(None)] * len(problem.state_shape))]
    right_states = paths.states[
        ..., index + 1, *([slice(None)] * len(problem.state_shape))
    ]
    transition_valid = paths.valid[..., index] & paths.valid[..., index + 1]
    state_mask = transition_valid.reshape(
        paths.sample_shape + (1,) * len(problem.state_shape)
    )
    safe_left_states = jnp.where(state_mask, left_states, 0.0)
    safe_right_states = jnp.where(state_mask, right_states, 0.0)
    value_key, control_key = jr.split(key)
    right_times = jnp.full(paths.sample_shape, right_time)
    next_values = _pointwise_values(
        next_value_predictor,
        right_times,
        safe_right_states,
        problem,
        key=value_key,
        output_shape=problem.output_shape,
    )
    source_controls = _pointwise_autodiff_control(
        next_value_predictor,
        right_times,
        safe_right_states,
        problem,
        key=control_key,
    )
    sample_count = prod(paths.sample_shape) if paths.sample_shape else 1

    def source_at(state, value, control):
        source = jnp.asarray(
            problem.generator(right_time, state, value, control, problem.args)
        )
        if source.shape != problem.output_shape:
            raise ValueError("BSDE generator returned an incompatible output shape.")
        return source

    source_values = jax.vmap(source_at)(
        safe_right_states.reshape((-1,) + problem.state_shape),
        next_values.reshape((-1,) + problem.output_shape),
        source_controls.reshape((-1,) + problem.output_shape + problem.noise_shape),
    ).reshape(paths.sample_shape + problem.output_shape)
    if source_values.shape[0 : len(paths.sample_shape)] != paths.sample_shape:
        raise RuntimeError(f"Failed to preserve {sample_count} splitting samples.")
    value_targets = jax.lax.stop_gradient(
        next_values + (right_time - left_time) * source_values
    )
    valid = (
        transition_valid
        & _event_finite(safe_left_states, problem.state_shape)
        & _event_finite(safe_right_states, problem.state_shape)
        & _event_finite(next_values, problem.output_shape)
        & _event_finite(source_controls, problem.output_shape + problem.noise_shape)
        & _event_finite(source_values, problem.output_shape)
        & _event_finite(value_targets, problem.output_shape)
    )
    return DeepSplittingLabelBatch(
        left_states=safe_left_states,
        right_states=safe_right_states,
        next_values=next_values,
        source_controls=source_controls,
        source_values=source_values,
        value_targets=value_targets,
        valid=valid,
        left_time=left_time,
        right_time=right_time,
        sample_shape=paths.sample_shape,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        slice_index=index,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        path_id=paths.path_id,
    )


DeepSplittingLabelProvider: TypeAlias = Callable[
    [Key[Array, ""]], DeepSplittingLabelBatch
]


class DeepSplittingRegressionTerm(AbstractSamplingTerm):
    """Supervised conditional-expectation regression term for one splitting slice."""

    problem: BSDEProblem
    fixed_labels: DeepSplittingLabelBatch | None
    label_provider: DeepSplittingLabelProvider | None
    value_weight: Array
    value_name: str = eqx.field(static=True)
    slice_index: int = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        problem: BSDEProblem,
        /,
        *,
        value_name: str,
        slice_index: int,
        labels: DeepSplittingLabelBatch | DeepSplittingLabelProvider,
        value_weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(problem, BSDEProblem):
            raise TypeError("problem must be a BSDEProblem.")
        if not isinstance(value_name, str) or not value_name:
            raise ValueError("value_name must be a non-empty string.")
        index = int(slice_index)
        if index < 0:
            raise ValueError("slice_index must be nonnegative.")
        if isinstance(labels, DeepSplittingLabelBatch):
            _validate_labels(labels, problem, index)
            fixed_labels = labels
            label_provider = None
        elif callable(labels):
            fixed_labels = None
            label_provider = labels
        else:
            raise TypeError("labels must be a DeepSplittingLabelBatch or provider.")
        weight = jnp.asarray(value_weight, dtype=float).reshape(())
        if bool(~jnp.isfinite(weight)) or float(weight) < 0.0:
            raise ValueError("value_weight must be finite and nonnegative.")
        self.problem = problem
        self.fixed_labels = fixed_labels
        self.label_provider = label_provider
        self.value_weight = weight
        self.value_name = value_name
        self.slice_index = index
        self.label = None if label is None else str(label)

    def sample(
        self,
        *,
        key: Key[Array, ""] = jr.key(0),
    ) -> DeepSplittingLabelBatch:
        if self.fixed_labels is not None:
            return self.fixed_labels
        if self.label_provider is None:
            raise RuntimeError("Deep splitting label provider is unavailable.")
        labels = self.label_provider(key)
        _validate_labels(labels, self.problem, self.slice_index)
        return labels

    def predictions(
        self,
        functions: Mapping[str, DomainFunction],
        labels: DeepSplittingLabelBatch,
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
    ) -> Array:
        _validate_labels(labels, self.problem, self.slice_index)
        if self.value_name not in functions:
            raise KeyError(f"Missing deep splitting value function {self.value_name!r}.")
        left_times = jnp.full(labels.sample_shape, labels.left_time)
        return _pointwise_values(
            functions[self.value_name],
            left_times,
            labels.left_states,
            self.problem,
            key=key,
            output_shape=self.problem.output_shape,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: DeepSplittingLabelBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        sampling_key, prediction_key = jr.split(key)
        labels = self.sample(key=sampling_key) if batch is None else batch
        predictions = self.predictions(functions, labels, key=prediction_key)
        return self.value_weight * _masked_mean_square(
            predictions - labels.value_targets,
            labels.valid,
            self.problem.output_shape,
        )

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: DeepSplittingLabelBatch | None = None,
    ) -> DeepSplittingRegressionDiagnostics:
        sampling_key, prediction_key = jr.split(key)
        labels = self.sample(key=sampling_key) if batch is None else batch
        predictions = self.predictions(functions, labels, key=prediction_key)
        one_step_rmse = _rms(
            predictions - labels.value_targets,
            labels.valid,
            self.problem.output_shape,
        )
        target_rms = _rms(
            labels.value_targets,
            labels.valid,
            self.problem.output_shape,
        )
        source_rms = _rms(
            labels.source_values,
            labels.valid,
            self.problem.output_shape,
        )
        finite = jnp.asarray(
            jnp.isfinite(one_step_rmse)
            & jnp.isfinite(target_rms)
            & jnp.isfinite(source_rms)
        )
        return DeepSplittingRegressionDiagnostics(
            one_step_rmse=one_step_rmse,
            target_rms=target_rms,
            source_rms=source_rms,
            valid_fraction=jnp.mean(labels.valid),
            finite=finite,
        )


__all__ = [
    "deep_splitting_labels",
    "DeepSplittingLabelBatch",
    "DeepSplittingLabelProvider",
    "DeepSplittingPredictor",
    "DeepSplittingRegressionDiagnostics",
    "DeepSplittingRegressionTerm",
]
