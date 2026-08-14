#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule
from ...stochastic._state_space import (
    AbstractTransitionKernel,
    StateSpaceStepContext,
    TransitionSample,
)


if TYPE_CHECKING:
    from ._solver import SchrodingerBridgeResult


class BridgePathSample(StrictModule):
    """Keyed exact bridge paths with explicit indices, validity, and provenance."""

    values: Array
    state_indices: Array
    valid: Array
    log_prob: Array
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)


def _flat_case_index(
    context: StateSpaceStepContext, case_shape: tuple[int, ...]
) -> Array:
    index = jnp.asarray(context.case_index, dtype=jnp.int32)
    if not case_shape:
        return jnp.asarray(0, dtype=jnp.int32)
    if index.shape == ():
        return index
    if index.shape != (len(case_shape),):
        raise ValueError(
            "context.case_index must be a flat scalar index or one index per case axis."
        )
    strides = []
    stride = 1
    for size in reversed(case_shape[1:]):
        stride *= size
        strides.append(stride)
    multipliers = jnp.asarray(tuple(reversed(strides)) + (1,), dtype=jnp.int32)
    return jnp.sum(index * multipliers)


def _state_indices(values: Array, support: Array, state_shape: tuple[int, ...], /):
    if state_shape:
        if (
            values.ndim < len(state_shape)
            or tuple(values.shape[-len(state_shape) :]) != state_shape
        ):
            raise ValueError(f"State values must end with state_shape {state_shape}.")
        batch_shape = values.shape[: -len(state_shape)]
        flat = values.reshape((-1,) + state_shape)
        comparison_axes = tuple(range(2, 2 + len(state_shape)))
        matches = jnp.all(flat[:, None, ...] == support[None, ...], axis=comparison_axes)
    else:
        batch_shape = values.shape
        flat = values.reshape((-1,))
        matches = flat[:, None] == support[None, :]
    valid = jnp.any(matches, axis=-1)
    indices = jnp.argmax(matches, axis=-1).astype(jnp.int32)
    return indices.reshape(batch_shape), valid.reshape(batch_shape)


class ControlledTransitionKernel(AbstractTransitionKernel):
    """Doob transform of a solved exact finite-state reference process."""

    support: Array
    times: Array
    probabilities: Array
    row_valid: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(self, result: SchrodingerBridgeResult, /):
        from ._solver import require_converged_bridge, SchrodingerBridgeResult

        if not isinstance(result, SchrodingerBridgeResult):
            raise TypeError("result must be a SchrodingerBridgeResult.")
        result = require_converged_bridge(result)
        self.support = result.problem.state_support
        self.times = result.problem.times
        self.probabilities = result.controlled_transition_probabilities
        self.row_valid = result.controlled_row_valid
        self.case_shape = result.problem.case_shape
        self.state_shape = result.problem.state_shape
        self.process_id = f"{result.problem.reference.process_id}:schrodinger-bridge"
        self.approximation_id = "exact-finite-state-doob"
        self.has_log_density = True

    @property
    def num_states(self) -> int:
        return int(self.support.shape[len(self.case_shape)])

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[0] - 1)

    def _case_step(self, context: StateSpaceStepContext, /) -> tuple[Array, Array]:
        case_index = _flat_case_index(context, self.case_shape)
        step_index = jnp.asarray(context.step_index, dtype=jnp.int32)
        case_index = eqx.error_if(
            case_index,
            (case_index < 0)
            | (case_index >= (prod(self.case_shape) if self.case_shape else 1)),
            "Bridge transition context.case_index is out of range.",
        )
        step_index = eqx.error_if(
            step_index,
            (step_index < 0) | (step_index >= self.num_steps),
            "Bridge transition context.step_index is out of range.",
        )
        return case_index, step_index

    def _validate_interval(
        self, step_index: Array, t0: ArrayLike, t1: ArrayLike, /
    ) -> Array:
        observed_start = jnp.asarray(t0)
        observed_end = jnp.asarray(t1)
        mismatch = (
            (observed_start.shape != ())
            | (observed_end.shape != ())
            | ~jnp.isclose(observed_start, self.times[step_index])
            | ~jnp.isclose(observed_end, self.times[step_index + 1])
        )
        return eqx.error_if(
            step_index,
            jnp.any(mismatch),
            "Bridge transition times do not match context.step_index and the solved grid.",
        )

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        case_index, step_index = self._case_step(context)
        step_index = self._validate_interval(step_index, t0, t1)
        count = prod(self.case_shape) if self.case_shape else 1
        support = self.support.reshape((count, self.num_states) + self.state_shape)[
            case_index
        ]
        probabilities = self.probabilities.reshape(
            (count, self.num_steps, self.num_states, self.num_states)
        )[case_index, step_index]
        state_array = jnp.asarray(state)
        indices, state_valid = _state_indices(state_array, support, self.state_shape)
        selected = probabilities[indices]
        log_probabilities = jnp.where(selected > 0.0, jnp.log(selected), -jnp.inf)
        draws = jr.categorical(key, log_probabilities, axis=-1).astype(jnp.int32)
        values = support[draws]
        valid = state_valid
        selector = valid.reshape(valid.shape + (1,) * len(self.state_shape))
        values = jnp.where(selector, values, jnp.full_like(values, jnp.nan))
        status = jnp.where(valid, 0, 1).astype(jnp.int32)
        return TransitionSample(
            values=values,
            valid=valid,
            status=status,
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        case_index, step_index = self._case_step(context)
        step_index = self._validate_interval(step_index, t0, t1)
        count = prod(self.case_shape) if self.case_shape else 1
        support = self.support.reshape((count, self.num_states) + self.state_shape)[
            case_index
        ]
        probabilities = self.probabilities.reshape(
            (count, self.num_steps, self.num_states, self.num_states)
        )[case_index, step_index]
        source_indices, source_valid = _state_indices(
            jnp.asarray(state), support, self.state_shape
        )
        target_indices, target_valid = _state_indices(
            jnp.asarray(next_state), support, self.state_shape
        )
        batch_shape = jnp.broadcast_shapes(source_indices.shape, target_indices.shape)
        source_indices = jnp.broadcast_to(source_indices, batch_shape)
        target_indices = jnp.broadcast_to(target_indices, batch_shape)
        valid = jnp.broadcast_to(source_valid, batch_shape) & jnp.broadcast_to(
            target_valid, batch_shape
        )
        probability = probabilities[source_indices, target_indices]
        return jnp.where(valid & (probability > 0.0), jnp.log(probability), -jnp.inf)


def _sample_indices_flat(
    key: Key[Array, ""], result: SchrodingerBridgeResult, sample_count: int, /
) -> Array:
    problem = result.problem
    case_count = problem.num_cases
    initial = problem.initial_probabilities.reshape((case_count, problem.num_states))
    controlled = result.controlled_transition_probabilities.reshape(
        (case_count, problem.num_steps, problem.num_states, problem.num_states)
    )

    def sample_case(case_index, initial_probability, transitions):
        case_key = jr.fold_in(key, case_index.astype(jnp.uint32))

        def sample_member(member_index):
            member_key = jr.fold_in(case_key, member_index.astype(jnp.uint32))
            start_key = jr.fold_in(member_key, jnp.asarray(0, dtype=jnp.uint32))
            initial_log = jnp.where(
                initial_probability > 0.0, jnp.log(initial_probability), -jnp.inf
            )
            initial_state = jr.categorical(start_key, initial_log).astype(jnp.int32)

            def step(state_index, step_data):
                step_index, matrix = step_data
                step_key = jr.fold_in(member_key, (step_index + 1).astype(jnp.uint32))
                probabilities = matrix[state_index]
                log_probabilities = jnp.where(
                    probabilities > 0.0, jnp.log(probabilities), -jnp.inf
                )
                next_index = jr.categorical(step_key, log_probabilities).astype(jnp.int32)
                return next_index, next_index

            _, following = jax.lax.scan(
                step,
                initial_state,
                (jnp.arange(problem.num_steps, dtype=jnp.int32), transitions),
            )
            return jnp.concatenate((initial_state[None], following), axis=0)

        return jax.vmap(sample_member)(jnp.arange(sample_count, dtype=jnp.int32))

    return jax.vmap(sample_case)(
        jnp.arange(case_count, dtype=jnp.int32), initial, controlled
    )


def sample_bridge_state_indices(
    key: Key[Array, ""],
    result: SchrodingerBridgeResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> Array:
    """Sample exact bridge indices with replay- and prefix-stable semantic keys."""
    from ._solver import require_converged_bridge, SchrodingerBridgeResult

    if not isinstance(result, SchrodingerBridgeResult):
        raise TypeError("result must be a SchrodingerBridgeResult.")
    result = require_converged_bridge(result)
    shape = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in shape):
        raise ValueError("sample_shape dimensions must be positive.")
    sample_count = prod(shape) if shape else 1
    indices = _sample_indices_flat(key, result, sample_count)
    trailing = shape + (result.problem.num_steps + 1,)
    if not shape:
        indices = indices[:, 0]
    return indices.reshape(result.problem.case_shape + trailing)


def sample_bridge_paths(
    key: Key[Array, ""],
    result: SchrodingerBridgeResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> Array:
    """Sample exact bridge state values with stable case/member/step keys."""
    indices = sample_bridge_state_indices(key, result, sample_shape=sample_shape)
    problem = result.problem
    sample_count = prod(sample_shape) if sample_shape else 1
    case_count = problem.num_cases
    flat_indices = indices.reshape((case_count, sample_count, problem.num_steps + 1))
    support = problem.state_support.reshape(
        (case_count, problem.num_states) + problem.state_shape
    )
    values = jax.vmap(lambda states, selected: states[selected])(support, flat_indices)
    if not sample_shape:
        values = values[:, 0]
    return values.reshape(
        problem.case_shape
        + tuple(sample_shape)
        + (problem.num_steps + 1,)
        + problem.state_shape
    )


def _path_indices(
    result: SchrodingerBridgeResult, paths: ArrayLike, /
) -> tuple[Array, Array, tuple[int, ...]]:
    problem = result.problem
    values = jnp.asarray(paths)
    event_rank = len(problem.state_shape)
    prefix = values.shape[: values.ndim - event_rank] if event_rank else values.shape
    if len(prefix) < len(problem.case_shape) + 1:
        raise ValueError("paths must contain declared case axes and one time axis.")
    if tuple(prefix[: len(problem.case_shape)]) != problem.case_shape:
        raise ValueError("paths must begin with the result case shape.")
    if prefix[-1] != problem.num_steps + 1:
        raise ValueError("paths must contain one state at every bridge grid time.")
    if event_rank and tuple(values.shape[-event_rank:]) != problem.state_shape:
        raise ValueError("paths must end with the bridge state shape.")
    sample_shape = tuple(int(size) for size in prefix[len(problem.case_shape) : -1])
    sample_count = prod(sample_shape) if sample_shape else 1
    case_count = problem.num_cases
    flat_values = values.reshape(
        (case_count, sample_count, problem.num_steps + 1) + problem.state_shape
    )
    support = problem.state_support.reshape(
        (case_count, problem.num_states) + problem.state_shape
    )

    def match_case(case_values, case_support):
        flattened = case_values.reshape(
            (sample_count * (problem.num_steps + 1),) + problem.state_shape
        )
        indices, valid = _state_indices(flattened, case_support, problem.state_shape)
        return indices.reshape((sample_count, problem.num_steps + 1)), valid.reshape(
            (sample_count, problem.num_steps + 1)
        )

    indices, valid = jax.vmap(match_case)(flat_values, support)
    return indices, valid, sample_shape


def _path_log_prob(
    result: SchrodingerBridgeResult, paths: ArrayLike, *, reference: bool
) -> Array:
    indices, valid, sample_shape = _path_indices(result, paths)
    problem = result.problem
    case_count = problem.num_cases
    initial = problem.initial_probabilities.reshape((case_count, problem.num_states))
    matrices = (
        jnp.exp(result.reference_log_transitions)
        if reference
        else result.controlled_transition_probabilities
    ).reshape((case_count, problem.num_steps, problem.num_states, problem.num_states))

    def evaluate_case(case_indices, case_valid, initial_probability, transitions):
        initial_values = initial_probability[case_indices[:, 0]]
        log_probability = jnp.where(
            initial_values > 0.0, jnp.log(initial_values), -jnp.inf
        )
        for step in range(problem.num_steps):
            values = transitions[step, case_indices[:, step], case_indices[:, step + 1]]
            log_probability = log_probability + jnp.where(
                values > 0.0, jnp.log(values), -jnp.inf
            )
        return jnp.where(jnp.all(case_valid, axis=-1), log_probability, -jnp.inf)

    values = jax.vmap(evaluate_case)(indices, valid, initial, matrices)
    if not sample_shape:
        values = values[:, 0]
    return values.reshape(problem.case_shape + sample_shape)


def bridge_path_log_prob(result: SchrodingerBridgeResult, paths: ArrayLike, /) -> Array:
    """Evaluate the solved controlled path law on finite-state paths."""
    return _path_log_prob(result, paths, reference=False)


def reference_path_log_prob(
    result: SchrodingerBridgeResult, paths: ArrayLike, /
) -> Array:
    """Evaluate the initial-endpoint reference path law on finite-state paths."""
    return _path_log_prob(result, paths, reference=True)


def sample_bridge(
    key: Key[Array, ""],
    result: SchrodingerBridgeResult,
    /,
    *,
    sample_shape: tuple[int, ...] = (),
) -> BridgePathSample:
    """Return keyed paths with fixed-structure scientific sampling diagnostics."""
    indices = sample_bridge_state_indices(key, result, sample_shape=sample_shape)
    values = sample_bridge_paths(key, result, sample_shape=sample_shape)
    log_prob = bridge_path_log_prob(result, values)
    valid = jnp.isfinite(log_prob)
    return BridgePathSample(
        values=values,
        state_indices=indices,
        valid=valid,
        log_prob=log_prob,
        process_id=f"{result.problem.reference.process_id}:schrodinger-bridge",
        approximation_id="exact-finite-state-doob",
    )


__all__ = [
    "BridgePathSample",
    "ControlledTransitionKernel",
    "bridge_path_log_prob",
    "reference_path_log_prob",
    "sample_bridge",
    "sample_bridge_paths",
    "sample_bridge_state_indices",
]
