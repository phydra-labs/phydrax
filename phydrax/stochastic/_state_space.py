#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._frozendict import frozendict
from .._strict import AbstractAttribute, StrictModule
from ._linear_gaussian import (
    degenerate_gaussian_log_prob,
    LinearGaussianDynamics,
    LinearGaussianParameterization,
    LinearGaussianParameters,
)
from ._process import AbstractMarginalTransitionLaw, AbstractProcessDistribution
from ._state_space_input import (
    AbstractStateSpaceInput,
    InputEvaluation,
)


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError(f"{owner} dimensions must be positive.")
    return resolved


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _names(value: Sequence[str], /, *, owner: str) -> tuple[str, ...]:
    resolved = tuple(str(name) for name in value)
    if any(not name for name in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError(f"{owner} must contain unique non-empty names.")
    return resolved


def _event_size(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


def _ends_with(array: Array, shape: tuple[int, ...], /, *, owner: str) -> None:
    if not shape:
        return
    if array.ndim < len(shape) or tuple(array.shape[-len(shape) :]) != shape:
        raise ValueError(f"{owner} must end in shape {shape}; got {array.shape}.")


def _event_finite(array: Array, event_shape: tuple[int, ...]) -> Array:
    if not event_shape:
        return jnp.isfinite(array)
    return jnp.all(
        jnp.isfinite(array), axis=tuple(range(array.ndim - len(event_shape), array.ndim))
    )


def state_space_key(
    root_key: Key[Array, ""],
    namespace: str,
    case_id: str,
    step: ArrayLike,
    /,
    *,
    member: ArrayLike = 0,
) -> Array:
    """Derive a stable state-space key from semantic identities, not batch position."""
    _name(namespace, owner="namespace")
    _name(case_id, owner="case_id")
    digest = hashlib.sha256(f"{namespace}\0{case_id}".encode()).digest()
    first = int.from_bytes(digest[:4], "little")
    second = int.from_bytes(digest[4:8], "little")
    key = jr.fold_in(root_key, first)
    key = jr.fold_in(key, second)
    key = jr.fold_in(key, jnp.asarray(step, dtype=jnp.uint32))
    return jr.fold_in(key, jnp.asarray(member, dtype=jnp.uint32))


class StateSpaceStepContext(StrictModule):
    """Dynamic context shared by one transition and observation schedule step."""

    args: Any
    case_index: Array
    step_index: Array
    transition_start_input: Array
    transition_end_input: Array
    observation_input: Array
    input_breakpoints: Array
    input_breakpoint_valid: Array
    input_valid: Array
    input_signal: AbstractStateSpaceInput | None

    @classmethod
    def empty(
        cls,
        /,
        *,
        args: Any = None,
        case_index: ArrayLike = 0,
        step_index: ArrayLike = 0,
    ) -> StateSpaceStepContext:
        absent = jnp.empty((0,), dtype=float)
        return cls(
            args=args,
            case_index=jnp.asarray(case_index, dtype=jnp.int32),
            step_index=jnp.asarray(step_index, dtype=jnp.int32),
            transition_start_input=absent,
            transition_end_input=absent,
            observation_input=absent,
            input_breakpoints=absent,
            input_breakpoint_valid=jnp.empty((0,), dtype=bool),
            input_valid=jnp.asarray(True),
            input_signal=None,
        )

    def evaluate_input(
        self,
        time: ArrayLike,
        /,
    ) -> InputEvaluation:
        """Evaluate the same interval-local input at an arbitrary internal time."""
        if self.input_signal is None:
            raise ValueError("This state-space step has no exogenous input signal.")
        return self.input_signal.evaluate(time, self.case_index)


class ObservationSequence(StrictModule):
    """Masked, axis-explicit observations over one or more physical cases."""

    times: Array
    values: Array
    step_valid: Array
    observation_mask: Array
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    observation_axes: tuple[str, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    sensor_id: str | None = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    approximation_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] = (),
        observation_axes: Sequence[str] = ("observation",),
        step_valid: ArrayLike | None = None,
        observation_mask: ArrayLike | None = None,
        case_ids: Sequence[str] | None = None,
        sequence_id: str = "observations",
        sensor_id: str | None = None,
        discretization_id: str | None = None,
        approximation_id: str | None = None,
    ):
        cases = _shape(case_shape, owner="case_shape")
        case_names = _names(case_axes, owner="case_axes")
        if len(cases) != len(case_names):
            raise ValueError("case_axes and case_shape must have equal rank.")
        observation_names = _names(observation_axes, owner="observation_axes")
        if set(case_names) & set(observation_names):
            raise ValueError("Case and observation axis names must be disjoint.")
        array = jnp.asarray(values)
        required_rank = len(cases) + 1 + len(observation_names)
        if array.ndim != required_rank or tuple(array.shape[: len(cases)]) != cases:
            raise ValueError(
                "values must have shape case_shape + (step,) + observation_shape."
            )
        step_position = len(cases)
        num_steps = int(array.shape[step_position])
        if num_steps <= 0:
            raise ValueError("Observation sequences require at least one step.")
        observation_shape = tuple(int(size) for size in array.shape[step_position + 1 :])
        if any(size <= 0 for size in observation_shape):
            raise ValueError("observation_shape dimensions must be positive.")
        time_array = jnp.asarray(times, dtype=float)
        if time_array.shape == (num_steps,):
            time_array = jnp.broadcast_to(time_array, cases + (num_steps,))
        if time_array.shape != cases + (num_steps,):
            raise ValueError("times must be a step vector or case-aligned step array.")
        if bool(jnp.any(~jnp.isfinite(time_array))):
            raise ValueError("Observation times must be finite, including padded steps.")
        if bool(jnp.any(~jnp.isfinite(array))):
            raise ValueError(
                "Observation values must be finite; use masks for missingness."
            )
        valid = (
            jnp.ones(cases + (num_steps,), dtype=bool)
            if step_valid is None
            else jnp.broadcast_to(
                jnp.asarray(step_valid, dtype=bool), cases + (num_steps,)
            )
        )
        if bool(jnp.any(valid[..., 1:] & ~valid[..., :-1])):
            raise ValueError("step_valid must be a prefix mask for every physical case.")
        if bool(jnp.any(~jnp.any(valid, axis=-1))):
            raise ValueError("Every physical case requires at least one valid step.")
        adjacent = valid[..., 1:] & valid[..., :-1]
        if bool(jnp.any(adjacent & (jnp.diff(time_array, axis=-1) <= 0.0))):
            raise ValueError("Valid observation times must be strictly increasing.")
        mask = (
            jnp.broadcast_to(
                valid.reshape(valid.shape + (1,) * len(observation_shape)), array.shape
            )
            if observation_mask is None
            else jnp.broadcast_to(jnp.asarray(observation_mask, dtype=bool), array.shape)
        )
        valid_expanded = valid.reshape(valid.shape + (1,) * len(observation_shape))
        if bool(jnp.any(mask & ~valid_expanded)):
            raise ValueError("Invalid padded steps cannot contain observed components.")
        count = prod(cases) if cases else 1
        ids = (
            tuple(f"case:{index}" for index in range(count))
            if case_ids is None
            else tuple(str(value) for value in case_ids)
        )
        if len(ids) != count or any(not value for value in ids) or len(set(ids)) != count:
            raise ValueError("case_ids must contain one unique non-empty ID per case.")
        for owner, value in (
            ("sensor_id", sensor_id),
            ("discretization_id", discretization_id),
            ("approximation_id", approximation_id),
        ):
            if value is not None:
                _name(value, owner=owner)
        self.times = time_array
        self.values = array
        self.step_valid = valid
        self.observation_mask = mask
        self.case_axes = case_names
        self.case_shape = cases
        self.observation_axes = observation_names
        self.observation_shape = observation_shape
        self.case_ids = ids
        self.sequence_id = _name(sequence_id, owner="sequence_id")
        self.sensor_id = sensor_id
        self.discretization_id = discretization_id
        self.approximation_id = approximation_id

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[-1])

    @property
    def num_cases(self) -> int:
        return prod(self.case_shape) if self.case_shape else 1


class AbstractStatePrior(StrictModule):
    """Initial-state law with explicit physical-case and state shapes."""

    state_shape: AbstractAttribute[tuple[int, ...]]
    batch_shape: AbstractAttribute[tuple[int, ...]]
    prior_id: AbstractAttribute[str]
    has_log_density: AbstractAttribute[bool]

    @property
    @abstractmethod
    def location(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError


class DistributionStatePrior(AbstractStatePrior):
    """Adapt an existing finite-dimensional process distribution as a state prior."""

    distribution: AbstractProcessDistribution
    state_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    prior_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(self, distribution: AbstractProcessDistribution, /, *, prior_id: str):
        if not isinstance(distribution, AbstractProcessDistribution):
            raise TypeError("distribution must implement AbstractProcessDistribution.")
        self.distribution = distribution
        self.state_shape = distribution.event_shape
        self.batch_shape = distribution.batch_shape
        self.prior_id = _name(prior_id, owner="prior_id")
        self.has_log_density = True

    @property
    def location(self) -> Array:
        return self.distribution.location

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        return self.distribution.sample(key, sample_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.distribution.log_prob(value)


class GaussianStatePrior(AbstractStatePrior):
    """Possibly singular Gaussian state prior with explicit covariance semantics."""

    mean: Array
    covariance: Array
    factor: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    prior_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        prior_id: str = "gaussian-prior",
    ):
        states = _shape(state_shape, owner="state_shape")
        size = _event_size(states)
        mean_array = jnp.asarray(mean, dtype=float)
        _ends_with(mean_array, states, owner="mean")
        batches = tuple(mean_array.shape[: -len(states)]) if states else mean_array.shape
        covariance_array = jnp.asarray(covariance, dtype=mean_array.dtype)
        expected = batches + (size, size)
        if covariance_array.shape == (size, size):
            covariance_array = jnp.broadcast_to(covariance_array, expected)
        if covariance_array.shape != expected:
            raise ValueError(f"covariance must have shape {(size, size)} or {expected}.")
        host = np.asarray(jax.device_get(covariance_array)).reshape((-1, size, size))
        if not np.all(np.isfinite(host)) or not np.allclose(
            host, np.swapaxes(host, -1, -2), atol=1e-10, rtol=1e-8
        ):
            raise ValueError("covariance must be finite and symmetric.")
        eigenvalues = np.linalg.eigvalsh(host)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("covariance must be positive semidefinite.")
        values, vectors = jnp.linalg.eigh(covariance_array)
        factor = vectors * jnp.sqrt(jnp.maximum(values, 0.0))[..., None, :]
        self.mean = mean_array
        self.covariance = covariance_array
        self.factor = factor
        self.state_shape = states
        self.batch_shape = tuple(int(size_) for size_ in batches)
        self.prior_id = _name(prior_id, owner="prior_id")
        self.has_log_density = bool(np.all(eigenvalues > 0.0))

    @property
    def location(self) -> Array:
        return self.mean

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        samples = _shape(sample_shape, owner="sample_shape")
        size = _event_size(self.state_shape)
        noise = jr.normal(
            key, samples + self.batch_shape + (size,), dtype=self.mean.dtype
        )
        values = jnp.einsum("...ij,...j->...i", self.factor, noise)
        mean = self.mean.reshape(self.batch_shape + (size,))
        return (values + mean).reshape(samples + self.batch_shape + self.state_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        if not self.has_log_density:
            raise ValueError("A singular Gaussian prior has no Lebesgue log density.")
        size = _event_size(self.state_shape)
        values = jnp.asarray(value, dtype=self.mean.dtype)
        expected = self.batch_shape + self.state_shape
        if values.shape != expected:
            raise ValueError(f"value must have shape {expected}; got {values.shape}.")
        residual = values.reshape(self.batch_shape + (size,)) - self.mean.reshape(
            self.batch_shape + (size,)
        )
        scale = jnp.linalg.cholesky(self.covariance)
        solved = jax.scipy.linalg.solve_triangular(
            scale, residual[..., None], lower=True
        )[..., 0]
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(scale, axis1=-2, axis2=-1)), axis=-1)
        return -0.5 * (
            jnp.sum(solved**2, axis=-1) + logdet + size * jnp.log(2.0 * jnp.pi)
        )


class CategoricalStatePrior(AbstractStatePrior):
    """Categorical law over explicit finite state values."""

    states: Array
    probabilities: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    prior_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        states: ArrayLike,
        probabilities: ArrayLike,
        /,
        *,
        prior_id: str = "categorical-prior",
    ):
        state_values = jnp.asarray(states)
        if state_values.ndim < 1 or state_values.shape[0] <= 0:
            raise ValueError("states must have one non-empty leading category axis.")
        probabilities_array = jnp.asarray(probabilities, dtype=float)
        categories = int(state_values.shape[0])
        if probabilities_array.ndim < 1 or probabilities_array.shape[-1] != categories:
            raise ValueError("probabilities must end in the state category count.")
        if bool(jnp.any(~jnp.isfinite(probabilities_array))) or bool(
            jnp.any(probabilities_array < 0.0)
        ):
            raise ValueError("probabilities must be finite and nonnegative.")
        total = jnp.sum(probabilities_array, axis=-1, keepdims=True)
        if bool(jnp.any(total <= 0.0)):
            raise ValueError("probabilities must have positive total mass.")
        self.states = state_values
        self.probabilities = probabilities_array / total
        self.state_shape = tuple(int(size) for size in state_values.shape[1:])
        self.batch_shape = tuple(int(size) for size in probabilities_array.shape[:-1])
        self.prior_id = _name(prior_id, owner="prior_id")
        self.has_log_density = True

    @property
    def location(self) -> Array:
        indices = jnp.argmax(self.probabilities, axis=-1)
        return self.states[indices]

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        samples = _shape(sample_shape, owner="sample_shape")
        logits = jnp.log(self.probabilities)
        indices = jr.categorical(
            key,
            logits,
            axis=-1,
            shape=samples + self.batch_shape,
        )
        return self.states[indices]

    def log_prob(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        expected = self.batch_shape + self.state_shape
        if values.shape != expected:
            raise ValueError(f"value must have shape {expected}; got {values.shape}.")
        categories = int(self.states.shape[0])
        if self.state_shape:
            left = values.reshape(self.batch_shape + (1,) + self.state_shape)
            right = self.states.reshape(
                (1,) * len(self.batch_shape) + (categories,) + self.state_shape
            )
            event_axes = tuple(
                range(
                    len(self.batch_shape) + 1,
                    len(self.batch_shape) + 1 + len(self.state_shape),
                )
            )
            matches = jnp.all(left == right, axis=event_axes)
        else:
            matches = values[..., None] == self.states
        probability = jnp.sum(self.probabilities * matches, axis=-1)
        return jnp.where(probability > 0.0, jnp.log(probability), -jnp.inf)


class TransitionSample(StrictModule):
    """One transition draw with explicit solver validity and status."""

    values: Array
    valid: Array
    status: Array
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)


class AbstractTransitionKernel(StrictModule):
    """Markov transition sampler with optional normalized transition density."""

    state_shape: AbstractAttribute[tuple[int, ...]]
    process_id: AbstractAttribute[str]
    approximation_id: AbstractAttribute[str]
    has_log_density: AbstractAttribute[bool]

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> TransitionSample:
        raise NotImplementedError

    @abstractmethod
    def log_prob(
        self,
        next_state: ArrayLike,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        raise NotImplementedError


class CallableTransitionKernel(AbstractTransitionKernel):
    sample_fn: Callable[
        [Array, Array, Array, Array, StateSpaceStepContext], Array | TransitionSample
    ]
    log_prob_fn: (
        Callable[[Array, Array, Array, Array, StateSpaceStepContext], Array] | None
    )
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        sample_fn: Callable[
            [Array, Array, Array, Array, StateSpaceStepContext],
            Array | TransitionSample,
        ],
        /,
        *,
        state_shape: Sequence[int],
        process_id: str,
        approximation_id: str,
        log_prob_fn: (
            Callable[[Array, Array, Array, Array, StateSpaceStepContext], Array] | None
        ) = None,
    ):
        if not callable(sample_fn):
            raise TypeError("sample_fn must be callable.")
        if log_prob_fn is not None and not callable(log_prob_fn):
            raise TypeError("log_prob_fn must be callable or None.")
        self.sample_fn = sample_fn
        self.log_prob_fn = log_prob_fn
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.process_id = _name(process_id, owner="process_id")
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.has_log_density = log_prob_fn is not None

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        state_array = jnp.asarray(state)
        _ends_with(state_array, self.state_shape, owner="state")
        result = self.sample_fn(
            key, state_array, jnp.asarray(t0), jnp.asarray(t1), context
        )
        if isinstance(result, TransitionSample):
            if result.process_id != self.process_id:
                raise ValueError("TransitionSample process_id does not match its kernel.")
            return result
        values = jnp.asarray(result)
        if values.shape != state_array.shape:
            raise ValueError("sample_fn must preserve the complete state batch shape.")
        valid = _event_finite(values, self.state_shape)
        return TransitionSample(
            values=values,
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        if self.log_prob_fn is None:
            raise ValueError("This transition kernel does not provide a log density.")
        return jnp.asarray(
            self.log_prob_fn(
                jnp.asarray(next_state),
                jnp.asarray(state),
                jnp.asarray(t0),
                jnp.asarray(t1),
                context,
            )
        )


class MarginalTransitionKernel(AbstractTransitionKernel):
    law: AbstractMarginalTransitionLaw
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        law: AbstractMarginalTransitionLaw,
        /,
        *,
        approximation_id: str = "marginal-transition",
    ):
        if not isinstance(law, AbstractMarginalTransitionLaw):
            raise TypeError("law must implement AbstractMarginalTransitionLaw.")
        self.law = law
        self.state_shape = law.state_shape
        self.process_id = _name(law.process_id, owner="law.process_id")
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.has_log_density = True

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        del context
        state_array = jnp.asarray(state)
        _ends_with(state_array, self.state_shape, owner="state")
        distribution = self.law.marginal_transition(state_array, t0=t0, t1=t1)
        values = jnp.asarray(distribution.sample(key))
        if values.shape != state_array.shape:
            raise ValueError(
                "Marginal transition samples must preserve state batch shape."
            )
        valid = _event_finite(values, self.state_shape)
        return TransitionSample(
            values=values,
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        del context
        distribution = self.law.marginal_transition(state, t0=t0, t1=t1)
        return distribution.log_prob(next_state)


def _parameter(value: Array | Callable[..., ArrayLike], *args: Array) -> Array:
    if callable(value):
        function = cast(Callable[..., ArrayLike], value)
        return jnp.asarray(function(*args), dtype=float)
    return jnp.asarray(value, dtype=float)


class LinearGaussianTransitionKernel(AbstractTransitionKernel):
    """Affine Gaussian transition driven by one interval parameterization."""

    parameterization: LinearGaussianParameterization | LinearGaussianDynamics
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        transition: (
            ArrayLike
            | Callable[[Array, Array], ArrayLike]
            | LinearGaussianParameterization
            | LinearGaussianDynamics
        ),
        covariance: ArrayLike | Callable[[Array, Array], ArrayLike] | None = None,
        /,
        *,
        state_shape: Sequence[int] | None = None,
        offset: ArrayLike | Callable[[Array, Array], ArrayLike] = 0.0,
        process_id: str | None = None,
        approximation_id: str | None = None,
        has_log_density: bool = True,
    ):
        if isinstance(
            transition, (LinearGaussianParameterization, LinearGaussianDynamics)
        ):
            if covariance is not None:
                raise TypeError(
                    "covariance must be omitted when using a parameterization object."
                )
            if callable(offset) or bool(jnp.any(jnp.asarray(offset) != 0.0)):
                raise TypeError(
                    "offset must be omitted when using a parameterization object."
                )
            parameterization = transition
            if (
                state_shape is not None
                and _shape(state_shape, owner="state_shape")
                != parameterization.state_shape
            ):
                raise ValueError(
                    "state_shape must match the parameterization state shape."
                )
            shape = parameterization.state_shape
            default_process_id = (
                parameterization.process_id
                if isinstance(parameterization, LinearGaussianDynamics)
                else "linear-gaussian"
            )
            default_approximation_id = (
                parameterization.approximation_id
                if isinstance(parameterization, LinearGaussianDynamics)
                else "exact-linear-gaussian"
            )
            parameterization_id = (
                parameterization.dynamics_id
                if isinstance(parameterization, LinearGaussianDynamics)
                else parameterization.parameterization_id
            )
        else:
            if covariance is None:
                raise TypeError(
                    "covariance is required for the legacy transition parameterization."
                )
            if state_shape is None:
                raise TypeError("state_shape is required.")
            shape = _shape(state_shape, owner="state_shape")
            parameterization = LinearGaussianParameterization(
                transition,
                covariance,
                state_shape=shape,
                offset=offset,
            )
            default_process_id = "linear-gaussian"
            default_approximation_id = "exact-linear-gaussian"
            parameterization_id = parameterization.parameterization_id
        self.parameterization = parameterization
        self.state_shape = shape
        self.process_id = _name(
            default_process_id if process_id is None else process_id,
            owner="process_id",
        )
        self.approximation_id = _name(
            (default_approximation_id if approximation_id is None else approximation_id),
            owner="approximation_id",
        )
        self.parameterization_id = parameterization_id
        self.resolved_method = parameterization.resolved_method
        self.has_log_density = bool(has_log_density)

    @property
    def transition(self):
        return (
            self.parameterization.transition
            if isinstance(self.parameterization, LinearGaussianParameterization)
            else self.parameterization
        )

    @property
    def offset(self):
        return self.parameterization.offset

    @property
    def covariance(self):
        return (
            self.parameterization.covariance
            if isinstance(self.parameterization, LinearGaussianParameterization)
            else self.parameterization.dispersion @ self.parameterization.dispersion.T
        )

    def parameters(
        self,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> LinearGaussianParameters:
        return self.parameterization.parameters(t0, t1, context)

    def mean(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        values = jnp.asarray(state)
        _ends_with(values, self.state_shape, owner="state")
        size = _event_size(self.state_shape)
        batch_shape = (
            values.shape[: -len(self.state_shape)] if self.state_shape else values.shape
        )
        parameters = self.parameters(t0, t1, context)
        flat = values.reshape(batch_shape + (size,))
        mean = (
            jnp.einsum("...ij,...j->...i", parameters.transition, flat)
            + parameters.offset
        )
        return mean.reshape(batch_shape + self.state_shape)

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        mean = self.mean(state, t0, t1, context)
        covariance = self.parameters(t0, t1, context).covariance
        size = _event_size(self.state_shape)
        batch_shape = (
            mean.shape[: -len(self.state_shape)] if self.state_shape else mean.shape
        )
        covariance = jnp.broadcast_to(covariance, batch_shape + (size, size))
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        tolerance = (
            jnp.finfo(covariance.dtype).eps
            * size
            * jnp.maximum(jnp.max(jnp.abs(eigenvalues), axis=-1), 1.0)
        )
        factor = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[..., None, :]
        noise = jr.normal(key, batch_shape + (size,), dtype=mean.dtype)
        draw = mean.reshape(batch_shape + (size,)) + jnp.einsum(
            "...ij,...j->...i", factor, noise
        )
        values = draw.reshape(mean.shape)
        valid = _event_finite(values, self.state_shape) & jnp.all(
            eigenvalues >= -tolerance[..., None], axis=-1
        )
        return TransitionSample(
            values=values,
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        if not self.has_log_density:
            raise ValueError("This transition kernel does not provide a log density.")
        mean = self.mean(state, t0, t1, context)
        covariance = self.parameters(t0, t1, context).covariance
        size = _event_size(self.state_shape)
        batch_shape = (
            mean.shape[: -len(self.state_shape)] if self.state_shape else mean.shape
        )
        covariance = jnp.broadcast_to(covariance, batch_shape + (size, size))
        residual = jnp.asarray(next_state).reshape(batch_shape + (size,)) - mean.reshape(
            batch_shape + (size,)
        )
        return degenerate_gaussian_log_prob(residual, covariance)


class AbstractObservationModel(StrictModule):
    """Observation location, normalized likelihood, and sampler."""

    state_shape: AbstractAttribute[tuple[int, ...]]
    observation_shape: AbstractAttribute[tuple[int, ...]]
    observation_id: AbstractAttribute[str]

    @abstractmethod
    def location(
        self, state: ArrayLike, time: ArrayLike, context: StateSpaceStepContext, /
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(
        self,
        value: ArrayLike,
        state: ArrayLike,
        time: ArrayLike,
        mask: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        state: ArrayLike,
        time: ArrayLike,
        context: StateSpaceStepContext,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        raise NotImplementedError


class CallableObservationModel(AbstractObservationModel):
    location_fn: Callable[[Array, Array, StateSpaceStepContext], Array]
    log_prob_fn: Callable[[Array, Array, Array, Array, StateSpaceStepContext], Array]
    sample_fn: Callable[
        [Array, Array, Array, tuple[int, ...], StateSpaceStepContext], Array
    ]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        location_fn: Callable[[Array, Array, StateSpaceStepContext], Array],
        log_prob_fn: Callable[[Array, Array, Array, Array, StateSpaceStepContext], Array],
        sample_fn: Callable[
            [Array, Array, Array, tuple[int, ...], StateSpaceStepContext], Array
        ],
        /,
        *,
        state_shape: Sequence[int],
        observation_shape: Sequence[int],
        observation_id: str,
    ):
        if (
            not callable(location_fn)
            or not callable(log_prob_fn)
            or not callable(sample_fn)
        ):
            raise TypeError("location_fn, log_prob_fn, and sample_fn must be callable.")
        self.location_fn = location_fn
        self.log_prob_fn = log_prob_fn
        self.sample_fn = sample_fn
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.observation_shape = _shape(observation_shape, owner="observation_shape")
        self.observation_id = _name(observation_id, owner="observation_id")

    def location(self, state, time, context, /) -> Array:
        values = jnp.asarray(
            self.location_fn(jnp.asarray(state), jnp.asarray(time), context)
        )
        _ends_with(values, self.observation_shape, owner="observation location")
        return values

    def log_prob(self, value, state, time, mask, context, /) -> Array:
        return jnp.asarray(
            self.log_prob_fn(
                jnp.asarray(value),
                jnp.asarray(state),
                jnp.asarray(time),
                jnp.asarray(mask),
                context,
            )
        )

    def sample(self, key, state, time, context, sample_shape=()) -> Array:
        return jnp.asarray(
            self.sample_fn(
                key,
                jnp.asarray(state),
                jnp.asarray(time),
                tuple(sample_shape),
                context,
            )
        )


def _masked_gaussian_log_prob(
    value: Array,
    location: Array,
    covariance: Array,
    mask: Array,
    /,
    *,
    observation_shape: tuple[int, ...],
) -> Array:
    size = _event_size(observation_shape)
    batch_shape = (
        location.shape[: -len(observation_shape)] if observation_shape else location.shape
    )
    value = jnp.broadcast_to(value, batch_shape + observation_shape).reshape(
        batch_shape + (size,)
    )
    location = location.reshape(batch_shape + (size,))
    mask = jnp.broadcast_to(mask, batch_shape + observation_shape).reshape(
        batch_shape + (size,)
    )
    covariance = jnp.broadcast_to(covariance, batch_shape + (size, size))
    active_pairs = mask[..., :, None] & mask[..., None, :]
    covariance = jnp.where(active_pairs, covariance, jnp.zeros_like(covariance))
    residual = jnp.where(mask, value - location, 0.0)
    return degenerate_gaussian_log_prob(residual, covariance)


class GaussianObservationModel(AbstractObservationModel):
    location_fn: Callable[[Array, Array, StateSpaceStepContext], Array]
    covariance: Array | Callable[[Array, StateSpaceStepContext], ArrayLike]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        location: Callable[[Array, Array, StateSpaceStepContext], Array],
        covariance: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike],
        /,
        *,
        state_shape: Sequence[int],
        observation_shape: Sequence[int],
        observation_id: str = "gaussian-observation",
    ):
        if not callable(location):
            raise TypeError("location must be callable.")
        self.location_fn = location
        self.covariance = (
            cast(Callable[[Array, StateSpaceStepContext], ArrayLike], covariance)
            if callable(covariance)
            else jnp.asarray(covariance, dtype=float)
        )
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.observation_shape = _shape(observation_shape, owner="observation_shape")
        self.observation_id = _name(observation_id, owner="observation_id")

    def location(self, state, time, context, /) -> Array:
        values = jnp.asarray(
            self.location_fn(jnp.asarray(state), jnp.asarray(time), context)
        )
        _ends_with(values, self.observation_shape, owner="observation location")
        return values

    def covariance_at(self, time: ArrayLike, context: StateSpaceStepContext, /) -> Array:
        values = _parameter(self.covariance, jnp.asarray(time), context)
        size = _event_size(self.observation_shape)
        if values.shape[-2:] != (size, size):
            raise ValueError("Observation covariance has an incompatible trailing shape.")
        return values

    def log_prob(self, value, state, time, mask, context, /) -> Array:
        return _masked_gaussian_log_prob(
            jnp.asarray(value),
            self.location(state, time, context),
            self.covariance_at(time, context),
            jnp.asarray(mask, dtype=bool),
            observation_shape=self.observation_shape,
        )

    def sample(self, key, state, time, context, sample_shape=()) -> Array:
        location = self.location(state, time, context)
        size = _event_size(self.observation_shape)
        batch_shape = (
            location.shape[: -len(self.observation_shape)]
            if self.observation_shape
            else location.shape
        )
        covariance = jnp.broadcast_to(
            self.covariance_at(time, context), batch_shape + (size, size)
        )
        scale = jnp.linalg.cholesky(covariance)
        samples = _shape(sample_shape, owner="sample_shape")
        noise = jr.normal(key, samples + batch_shape + (size,), dtype=location.dtype)
        values = location.reshape(batch_shape + (size,)) + jnp.einsum(
            "...ij,...j->...i", scale, noise
        )
        return values.reshape(samples + batch_shape + self.observation_shape)


class LinearGaussianObservationModel(AbstractObservationModel):
    matrix: Array | Callable[[Array, StateSpaceStepContext], ArrayLike]
    offset: Array | Callable[[Array, StateSpaceStepContext], ArrayLike]
    covariance: Array | Callable[[Array, StateSpaceStepContext], ArrayLike]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike],
        covariance: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike],
        /,
        *,
        state_shape: Sequence[int],
        observation_shape: Sequence[int],
        offset: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike] = 0.0,
        observation_id: str = "linear-gaussian-observation",
    ):
        self.matrix = (
            cast(Callable[[Array, StateSpaceStepContext], ArrayLike], matrix)
            if callable(matrix)
            else jnp.asarray(matrix, dtype=float)
        )
        self.offset = (
            cast(Callable[[Array, StateSpaceStepContext], ArrayLike], offset)
            if callable(offset)
            else jnp.asarray(offset, dtype=float)
        )
        self.covariance = (
            cast(Callable[[Array, StateSpaceStepContext], ArrayLike], covariance)
            if callable(covariance)
            else jnp.asarray(covariance, dtype=float)
        )
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.observation_shape = _shape(observation_shape, owner="observation_shape")
        self.observation_id = _name(observation_id, owner="observation_id")

    def parameters(
        self, time: ArrayLike, context: StateSpaceStepContext, /
    ) -> tuple[Array, Array, Array]:
        t = jnp.asarray(time)
        state_size = _event_size(self.state_shape)
        observation_size = _event_size(self.observation_shape)
        matrix = _parameter(self.matrix, t, context)
        covariance = _parameter(self.covariance, t, context)
        offset = _parameter(self.offset, t, context)
        if matrix.shape[-2:] != (observation_size, state_size):
            raise ValueError("Observation matrix has an incompatible trailing shape.")
        if covariance.shape[-2:] != (observation_size, observation_size):
            raise ValueError("Observation covariance has an incompatible trailing shape.")
        offset = jnp.broadcast_to(offset, matrix.shape[:-2] + (observation_size,))
        return matrix, offset, covariance

    def location(self, state, time, context, /) -> Array:
        state_array = jnp.asarray(state, dtype=float)
        _ends_with(state_array, self.state_shape, owner="state")
        state_size = _event_size(self.state_shape)
        observation_size = _event_size(self.observation_shape)
        batch_shape = (
            state_array.shape[: -len(self.state_shape)]
            if self.state_shape
            else state_array.shape
        )
        matrix, offset, _ = self.parameters(time, context)
        values = (
            jnp.einsum(
                "...ij,...j->...i",
                matrix,
                state_array.reshape(batch_shape + (state_size,)),
            )
            + offset
        )
        return values.reshape(batch_shape + self.observation_shape)

    def log_prob(self, value, state, time, mask, context, /) -> Array:
        _, _, covariance = self.parameters(time, context)
        return _masked_gaussian_log_prob(
            jnp.asarray(value),
            self.location(state, time, context),
            covariance,
            jnp.asarray(mask, dtype=bool),
            observation_shape=self.observation_shape,
        )

    def sample(self, key, state, time, context, sample_shape=()) -> Array:
        location = self.location(state, time, context)
        _, _, covariance = self.parameters(time, context)
        observation_size = _event_size(self.observation_shape)
        batch_shape = (
            location.shape[: -len(self.observation_shape)]
            if self.observation_shape
            else location.shape
        )
        covariance = jnp.broadcast_to(
            covariance, batch_shape + (observation_size, observation_size)
        )
        scale = jnp.linalg.cholesky(covariance)
        samples = _shape(sample_shape, owner="sample_shape")
        noise = jr.normal(
            key, samples + batch_shape + (observation_size,), dtype=location.dtype
        )
        values = location.reshape(batch_shape + (observation_size,)) + jnp.einsum(
            "...ij,...j->...i", scale, noise
        )
        return values.reshape(samples + batch_shape + self.observation_shape)


class StateSpaceModel(StrictModule):
    """Composable prior, transition, and observation roles for one hidden state."""

    prior: AbstractStatePrior
    transition: AbstractTransitionKernel
    observation: AbstractObservationModel
    metadata: frozendict[str, Any]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    parameter_id: str | None = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        prior: AbstractStatePrior,
        transition: AbstractTransitionKernel,
        observation: AbstractObservationModel,
        /,
        *,
        model_id: str,
        parameter_id: str | None = None,
        basis_id: str | None = None,
        discretization_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(prior, AbstractStatePrior):
            raise TypeError("prior must implement AbstractStatePrior.")
        if not isinstance(transition, AbstractTransitionKernel):
            raise TypeError("transition must implement AbstractTransitionKernel.")
        if not isinstance(observation, AbstractObservationModel):
            raise TypeError("observation must implement AbstractObservationModel.")
        if (
            prior.state_shape != transition.state_shape
            or prior.state_shape != observation.state_shape
        ):
            raise ValueError(
                "Prior, transition, and observation state shapes must agree."
            )
        for owner, value in (
            ("parameter_id", parameter_id),
            ("basis_id", basis_id),
            ("discretization_id", discretization_id),
        ):
            if value is not None:
                _name(value, owner=owner)
        self.prior = prior
        self.transition = transition
        self.observation = observation
        self.metadata = frozendict({} if metadata is None else metadata)
        self.state_shape = prior.state_shape
        self.observation_shape = observation.observation_shape
        self.model_id = _name(model_id, owner="model_id")
        self.parameter_id = parameter_id
        self.basis_id = basis_id
        self.discretization_id = discretization_id
        self.approximation_id = transition.approximation_id


def _state_space_input_validity(
    observations: ObservationSequence,
    initial_time: Array,
    input_signal: AbstractStateSpaceInput | None,
    /,
) -> Array:
    """Prevalidate one typed input over every active physical schedule interval."""
    num_steps = observations.num_steps
    if input_signal is None:
        return jnp.ones(observations.case_shape + (num_steps,), dtype=bool)
    case_count = prod(observations.case_shape) if observations.case_shape else 1
    flat_times = observations.times.reshape((case_count, num_steps))
    flat_initial = initial_time.reshape((case_count,))
    flat_active = observations.step_valid.reshape((case_count, num_steps))
    case_validity: list[Array] = []
    for case_index in range(case_count):
        step_validity: list[Array] = []
        for step_index in range(num_steps):
            start = (
                flat_initial[case_index]
                if step_index == 0
                else flat_times[case_index, step_index - 1]
            )
            end = flat_times[case_index, step_index]
            start_evaluation = input_signal.evaluate(start, case_index)
            end_evaluation = input_signal.evaluate(end, case_index)
            valid = start_evaluation.valid & end_evaluation.valid
            step_validity.append(
                jnp.where(flat_active[case_index, step_index], valid, True)
            )
        case_validity.append(jnp.stack(step_validity))
    input_valid = jnp.stack(case_validity).reshape(observations.case_shape + (num_steps,))
    if bool(jnp.any(~input_valid)):
        raise ValueError(
            "Input signal does not support every active transition endpoint "
            "and observation time."
        )
    return input_valid


def _state_space_step_context(
    observations: ObservationSequence,
    initial_time: Array,
    input_signal: AbstractStateSpaceInput | None,
    input_valid: Array,
    args: Any,
    case_index: ArrayLike,
    step_index: ArrayLike,
    /,
) -> StateSpaceStepContext:
    """Construct one canonical context from a validated physical schedule."""
    case_index_ = jnp.asarray(case_index, dtype=jnp.int32).reshape(())
    step_index_ = jnp.asarray(step_index, dtype=jnp.int32).reshape(())
    if input_signal is None:
        return StateSpaceStepContext.empty(
            args=args,
            case_index=case_index_,
            step_index=step_index_,
        )

    case_count = prod(observations.case_shape) if observations.case_shape else 1
    num_steps = observations.num_steps
    times = observations.times.reshape((case_count, num_steps))
    initial = initial_time.reshape((case_count,))
    completed = step_index_ >= num_steps
    schedule_index = jnp.clip(step_index_, 0, num_steps - 1)
    previous_index = jnp.maximum(schedule_index - 1, 0)
    end = times[case_index_, schedule_index]
    start = jnp.where(
        schedule_index == 0,
        initial[case_index_],
        times[case_index_, previous_index],
    )
    start = jnp.where(completed, end, start)
    start_evaluation = input_signal.evaluate(start, case_index_)
    end_evaluation = input_signal.evaluate(end, case_index_)
    input_breakpoints, input_breakpoint_valid = input_signal.breakpoints(
        start,
        end,
        case_index_,
    )
    schedule_valid = input_valid.reshape((case_count, num_steps))[
        case_index_, schedule_index
    ]
    valid = jnp.where(
        completed,
        end_evaluation.valid,
        schedule_valid & start_evaluation.valid & end_evaluation.valid,
    )
    return StateSpaceStepContext(
        args=args,
        case_index=case_index_,
        step_index=step_index_,
        transition_start_input=start_evaluation.value,
        transition_end_input=end_evaluation.value,
        observation_input=end_evaluation.value,
        input_breakpoints=input_breakpoints,
        input_breakpoint_valid=input_breakpoint_valid,
        input_valid=valid,
        input_signal=input_signal,
    )


class StateSpaceProblem(StrictModule):
    """State-space model bound to one canonical masked observation schedule."""

    model: StateSpaceModel
    observations: ObservationSequence
    initial_time: Array
    input_signal: AbstractStateSpaceInput | None
    input_valid: Array
    args: Any
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: StateSpaceModel,
        observations: ObservationSequence,
        /,
        *,
        initial_time: ArrayLike,
        problem_id: str,
        args: Any = None,
        input_signal: AbstractStateSpaceInput | None = None,
    ):
        if not isinstance(model, StateSpaceModel):
            raise TypeError("model must be a StateSpaceModel.")
        if not isinstance(observations, ObservationSequence):
            raise TypeError("observations must be an ObservationSequence.")
        if model.prior.batch_shape != observations.case_shape:
            raise ValueError("Prior batch_shape must equal the observation case_shape.")
        if model.observation_shape != observations.observation_shape:
            raise ValueError("Model and sequence observation shapes must agree.")
        if input_signal is not None:
            if not isinstance(input_signal, AbstractStateSpaceInput):
                raise TypeError("input_signal must implement AbstractStateSpaceInput.")
            if input_signal.case_shape != observations.case_shape:
                raise ValueError(
                    "Input signal case_shape must equal the observation case_shape."
                )
        initial = jnp.asarray(initial_time, dtype=float)
        initial = jnp.broadcast_to(initial, observations.case_shape)
        if bool(jnp.any(~jnp.isfinite(initial))):
            raise ValueError("initial_time must be finite.")
        first = observations.times[..., 0]
        if bool(jnp.any(initial > first)):
            raise ValueError(
                "initial_time cannot exceed the first valid observation time."
            )
        input_valid = _state_space_input_validity(
            observations,
            initial,
            input_signal,
        )
        self.model = model
        self.observations = observations
        self.initial_time = initial
        self.input_signal = input_signal
        self.input_valid = input_valid
        self.args = args
        self.problem_id = _name(problem_id, owner="problem_id")

    def step_context(
        self,
        case_index: ArrayLike,
        step_index: ArrayLike,
        /,
    ) -> StateSpaceStepContext:
        """Construct the canonical context for one physical schedule step."""
        return _state_space_step_context(
            self.observations,
            self.initial_time,
            self.input_signal,
            self.input_valid,
            self.args,
            case_index,
            step_index,
        )


__all__ = [
    "AbstractObservationModel",
    "AbstractStatePrior",
    "AbstractTransitionKernel",
    "CallableObservationModel",
    "CallableTransitionKernel",
    "CategoricalStatePrior",
    "DistributionStatePrior",
    "GaussianObservationModel",
    "GaussianStatePrior",
    "LinearGaussianObservationModel",
    "LinearGaussianTransitionKernel",
    "MarginalTransitionKernel",
    "ObservationSequence",
    "state_space_key",
    "StateSpaceModel",
    "StateSpaceProblem",
    "StateSpaceStepContext",
    "TransitionSample",
]
