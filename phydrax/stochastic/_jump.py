#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Callable, Sequence
from math import isfinite, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import AbstractAttribute, StrictModule


JumpStatus: TypeAlias = Literal[
    "success",
    "max_events",
    "invalid_intensity",
    "solver_failure",
]
JumpSide: TypeAlias = Literal["left", "right"]

JUMP_SUCCESS = 0
JUMP_MAX_EVENTS = 1
JUMP_INVALID_INTENSITY = 2
JUMP_SOLVER_FAILURE = 3


def jump_status_name(code: int, /) -> JumpStatus:
    value = int(code)
    if value == JUMP_SUCCESS:
        return "success"
    if value == JUMP_MAX_EVENTS:
        return "max_events"
    if value == JUMP_INVALID_INTENSITY:
        return "invalid_intensity"
    if value == JUMP_SOLVER_FAILURE:
        return "solver_failure"
    raise ValueError(f"Unknown jump status code {value}.")


def _hash_array(digest: hashlib._Hash, value: ArrayLike, /) -> None:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _hash_parts(prefix: bytes, *parts: Any) -> str:
    digest = hashlib.sha256(prefix)
    for part in parts:
        if isinstance(part, (jax.Array, np.ndarray)):
            _hash_array(digest, part)
        else:
            digest.update(repr(part).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _key(value: Key[Array, ""], /, *, owner: str) -> Array:
    data = jr.key_data(value)
    if data.shape != (2,):
        raise ValueError(f"{owner} requires one scalar JAX PRNG key.")
    return value


def _positive_shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _support(value: tuple[float, float], /, *, owner: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{owner} support must contain two bounds.")
    start, end = (float(bound) for bound in value)
    if not isfinite(start) or not isfinite(end) or not end > start:
        raise ValueError(f"{owner} support requires finite bounds with end > start.")
    return start, end


class PoissonClockRealization(StrictModule):
    """Prefix-stable unit-rate Poisson clocks for finite-activity channels.

    Physical event times are obtained by crossing these internal-time thresholds with
    integrated channel intensities. Every threshold and mark key is derived from the
    tuple ``(path, channel, event)`` so growing either batch or event capacity preserves
    every existing prefix exactly.
    """

    root_key: Array
    path_indices: Array
    support: tuple[float, float] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    max_events_per_channel: int = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_key: Key[Array, ""],
        num_channels: int,
        /,
        *,
        support: tuple[float, float],
        max_events_per_channel: int,
        sample_shape: Sequence[int] = (),
        process_id: str,
        label: str | None = None,
        coupling_id: str | None = None,
        _path_indices: Array | None = None,
    ):
        key = _key(root_key, owner="PoissonClockRealization")
        channels = int(num_channels)
        capacity = int(max_events_per_channel)
        if channels <= 0:
            raise ValueError("num_channels must be positive.")
        if capacity <= 0:
            raise ValueError("max_events_per_channel must be positive.")
        samples = _positive_shape(sample_shape, owner="sample_shape")
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("process_id must be a non-empty string.")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be a non-empty string or None.")
        if coupling_id is not None and (
            not isinstance(coupling_id, str) or not coupling_id
        ):
            raise ValueError("coupling_id must be a non-empty string or None.")
        expected = samples
        if _path_indices is None:
            count = prod(samples) if samples else 1
            indices = jnp.arange(count, dtype=jnp.uint32).reshape(expected)
        else:
            indices = jnp.asarray(_path_indices, dtype=jnp.uint32)
            if tuple(indices.shape) != expected:
                raise ValueError("path indices must match sample_shape.")
        support_value = _support(support, owner="PoissonClockRealization")
        resolved_coupling = coupling_id or _hash_parts(
            b"phydrax-poisson-coupling\0",
            jr.key_data(key),
            support_value,
            channels,
            process_id,
        )
        realization_id = _hash_parts(
            b"phydrax-poisson-realization\0",
            jr.key_data(key),
            support_value,
            channels,
            capacity,
            samples,
            indices,
            process_id,
        )
        self.root_key = key
        self.path_indices = indices
        self.support = support_value
        self.num_channels = channels
        self.max_events_per_channel = capacity
        self.sample_shape = samples
        self.process_id = process_id
        self.label = label
        self.realization_id = realization_id
        self.coupling_id = resolved_coupling

    @property
    def num_paths(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1

    @property
    def path_keys(self) -> Array:
        flat = self.path_indices.reshape((-1,))
        keys = jax.vmap(lambda index: jr.fold_in(self.root_key, index))(flat)
        return keys.reshape(self.sample_shape + tuple(self.root_key.shape))

    def _event_keys(self, namespace: int, /) -> Array:
        channels = jnp.arange(self.num_channels, dtype=jnp.uint32)
        events = jnp.arange(self.max_events_per_channel, dtype=jnp.uint32)
        flat_paths = self.path_keys.reshape((-1,) + tuple(self.root_key.shape))

        def one_path(path_key: Array) -> Array:
            namespaced = jr.fold_in(path_key, namespace)
            channel_keys = jax.vmap(lambda index: jr.fold_in(namespaced, index))(channels)
            return jax.vmap(
                lambda channel_key: jax.vmap(
                    lambda index: jr.fold_in(channel_key, index)
                )(events)
            )(channel_keys)

        keys = jax.vmap(one_path)(flat_paths)
        return keys.reshape(
            self.sample_shape
            + (self.num_channels, self.max_events_per_channel)
            + tuple(self.root_key.shape)
        )

    @property
    def threshold_increments(self) -> Array:
        keys = self._event_keys(0)
        flat = keys.reshape((-1,) + tuple(self.root_key.shape))
        values = jax.vmap(lambda key: jr.exponential(key))(flat)
        return values.reshape(
            self.sample_shape + (self.num_channels, self.max_events_per_channel)
        )

    @property
    def thresholds(self) -> Array:
        """Strictly increasing internal-time thresholds for each channel."""
        return jnp.cumsum(self.threshold_increments, axis=-1)

    @property
    def mark_keys(self) -> Array:
        """Stable event mark keys aligned with ``thresholds``."""
        return self._event_keys(1)

    @property
    def direct_event_keys(self) -> Array:
        """Prefix-stable keys for total-rate direct SSA event proposals."""
        events = jnp.arange(
            self.num_channels * self.max_events_per_channel,
            dtype=jnp.uint32,
        )
        flat_paths = self.path_keys.reshape((-1,) + tuple(self.root_key.shape))

        def one_path(path_key: Array) -> Array:
            namespaced = jr.fold_in(path_key, 2)
            return jax.vmap(lambda index: jr.fold_in(namespaced, index))(events)

        keys = jax.vmap(one_path)(flat_paths)
        return keys.reshape(
            self.sample_shape
            + (self.num_channels * self.max_events_per_channel,)
            + tuple(self.root_key.shape)
        )

    def extend(self, max_events_per_channel: int, /) -> PoissonClockRealization:
        """Increase event capacity while preserving all existing thresholds."""
        capacity = int(max_events_per_channel)
        if capacity < self.max_events_per_channel:
            raise ValueError("Extended capacity cannot be smaller than current capacity.")
        return PoissonClockRealization(
            self.root_key,
            self.num_channels,
            support=self.support,
            max_events_per_channel=capacity,
            sample_shape=self.sample_shape,
            process_id=self.process_id,
            label=self.label,
            coupling_id=self.coupling_id,
            _path_indices=self.path_indices,
        )


class JumpEventBatch(StrictModule):
    """Masked, fixed-capacity event streams with explicit termination status."""

    times: Array
    channels: Array
    marks: Array
    valid: Array
    status: Array
    pre_states: Array | None
    post_states: Array | None
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    max_events: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        channels: ArrayLike,
        marks: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        /,
        *,
        mark_shape: Sequence[int] = (),
        state_shape: Sequence[int] = (),
        pre_states: ArrayLike | None = None,
        post_states: ArrayLike | None = None,
    ):
        time_values = jnp.asarray(times, dtype=float)
        if time_values.ndim < 1 or time_values.shape[-1] <= 0:
            raise ValueError("times must have a non-empty trailing event axis.")
        batch = tuple(int(size) for size in time_values.shape[:-1])
        capacity = int(time_values.shape[-1])
        channel_values = jnp.asarray(channels, dtype=jnp.int32)
        valid_values = jnp.asarray(valid, dtype=bool)
        if (
            channel_values.shape != time_values.shape
            or valid_values.shape != time_values.shape
        ):
            raise ValueError("times, channels, and valid must have equal shapes.")
        status_values = jnp.asarray(status, dtype=jnp.int32)
        if status_values.shape != batch:
            raise ValueError("status must have shape batch_shape.")
        marks_shape = tuple(int(size) for size in mark_shape)
        if any(size <= 0 for size in marks_shape):
            raise ValueError("mark_shape dimensions must be positive.")
        mark_values = jnp.asarray(marks)
        if mark_values.shape != batch + (capacity,) + marks_shape:
            raise ValueError(
                "marks must have shape batch_shape + (max_events,) + mark_shape."
            )
        states_shape = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in states_shape):
            raise ValueError("state_shape dimensions must be positive.")
        expected_states = batch + (capacity,) + states_shape
        before = None if pre_states is None else jnp.asarray(pre_states)
        after = None if post_states is None else jnp.asarray(post_states)
        if (before is None) != (after is None):
            raise ValueError("pre_states and post_states must be supplied together.")
        if (
            before is not None
            and after is not None
            and (before.shape != expected_states or after.shape != expected_states)
        ):
            raise ValueError("Event states have incompatible shapes.")
        self.times = time_values
        self.channels = channel_values
        self.marks = mark_values
        self.valid = valid_values
        self.status = status_values
        self.pre_states = before
        self.post_states = after
        self.batch_shape = batch
        self.max_events = capacity
        self.mark_shape = marks_shape
        self.state_shape = states_shape

    @property
    def counts(self) -> Array:
        return jnp.sum(self.valid, axis=-1)

    @property
    def successful(self) -> Array:
        return self.status == JUMP_SUCCESS

    @property
    def overflow(self) -> Array:
        return self.status == JUMP_MAX_EVENTS

    def states_at(
        self,
        query_times: ArrayLike,
        initial_state: ArrayLike,
        /,
        *,
        side: JumpSide = "right",
    ) -> Array:
        """Evaluate the piecewise-constant event trajectory at query times."""
        if self.post_states is None:
            raise ValueError("states_at requires stored pre_states and post_states.")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        queries = jnp.asarray(query_times, dtype=self.times.dtype)
        if queries.ndim == 1:
            queries = jnp.broadcast_to(queries, self.batch_shape + queries.shape)
        if queries.shape[:-1] != self.batch_shape:
            raise ValueError("query_times must be a vector or batch-aligned vectors.")
        initial = jnp.asarray(initial_state, dtype=self.post_states.dtype)
        initial = jnp.broadcast_to(initial, self.batch_shape + self.state_shape)
        flat_times = self.times.reshape((-1, self.max_events))
        flat_valid = self.valid.reshape((-1, self.max_events))
        flat_states = self.post_states.reshape((-1, self.max_events) + self.state_shape)
        flat_queries = queries.reshape((-1, queries.shape[-1]))
        flat_initial = initial.reshape((-1,) + self.state_shape)

        def one_path(
            event_times: Array,
            event_valid: Array,
            event_states: Array,
            path_queries: Array,
            path_initial: Array,
        ) -> Array:
            if side == "right":
                eligible = event_valid[:, None] & (
                    event_times[:, None] <= path_queries[None, :]
                )
            else:
                eligible = event_valid[:, None] & (
                    event_times[:, None] < path_queries[None, :]
                )
            indices = jnp.sum(eligible, axis=0).astype(jnp.int32) - 1
            selected = event_states[jnp.maximum(indices, 0)]
            prefix = (indices >= 0).reshape(
                (indices.shape[0],) + (1,) * len(self.state_shape)
            )
            return jnp.where(prefix, selected, path_initial)

        values = jax.vmap(one_path)(
            flat_times,
            flat_valid,
            flat_states,
            flat_queries,
            flat_initial,
        )
        return values.reshape(self.batch_shape + (queries.shape[-1],) + self.state_shape)


class AbstractJumpProcess(StrictModule):
    """Finite-activity jump mechanism independent of its numerical solver."""

    state_shape: AbstractAttribute[tuple[int, ...]]
    num_channels: AbstractAttribute[int]
    mark_shape: AbstractAttribute[tuple[int, ...]]
    process_id: AbstractAttribute[str]

    @abstractmethod
    def intensities(self, t: ArrayLike, state: ArrayLike, args: Any = None, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def jump(
        self,
        state: ArrayLike,
        channel: ArrayLike,
        mark: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample_mark(
        self,
        key: Key[Array, ""],
        t: ArrayLike,
        state: ArrayLike,
        channel: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError


IntensityFunction = Callable[[ArrayLike, ArrayLike, Any], ArrayLike]
JumpMap = Callable[[ArrayLike, ArrayLike, ArrayLike, Any], ArrayLike]
MarkSampler = Callable[[Key[Array, ""], ArrayLike, ArrayLike, ArrayLike, Any], ArrayLike]


class JumpProcess(AbstractJumpProcess):
    """Callable-defined finite-activity jump process."""

    intensity_fn: IntensityFunction
    jump_fn: JumpMap
    mark_fn: MarkSampler | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        intensity_fn: IntensityFunction,
        jump_fn: JumpMap,
        /,
        *,
        state_shape: Sequence[int],
        num_channels: int,
        process_id: str,
        mark_shape: Sequence[int] = (),
        mark_fn: MarkSampler | None = None,
    ):
        if not callable(intensity_fn) or not callable(jump_fn):
            raise TypeError("intensity_fn and jump_fn must be callable.")
        if mark_fn is not None and not callable(mark_fn):
            raise TypeError("mark_fn must be callable or None.")
        states = tuple(int(size) for size in state_shape)
        if any(size <= 0 for size in states):
            raise ValueError("state_shape dimensions must be positive.")
        channels = int(num_channels)
        if channels <= 0:
            raise ValueError("num_channels must be positive.")
        marks = tuple(int(size) for size in mark_shape)
        if any(size <= 0 for size in marks):
            raise ValueError("mark_shape dimensions must be positive.")
        if marks and mark_fn is None:
            raise ValueError("A non-empty mark_shape requires mark_fn.")
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("process_id must be a non-empty string.")
        self.intensity_fn = intensity_fn
        self.jump_fn = jump_fn
        self.mark_fn = mark_fn
        self.state_shape = states
        self.num_channels = channels
        self.mark_shape = marks
        self.process_id = process_id

    def intensities(self, t: ArrayLike, state: ArrayLike, args: Any = None, /) -> Array:
        values = jnp.asarray(self.intensity_fn(t, state, args), dtype=float)
        if values.shape[-1:] != (self.num_channels,):
            raise ValueError("intensity_fn must return a trailing channel axis.")
        return values

    def jump(
        self,
        state: ArrayLike,
        channel: ArrayLike,
        mark: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        state_values = jnp.asarray(state)
        values = jnp.asarray(self.jump_fn(state_values, channel, mark, args))
        if values.shape != state_values.shape:
            raise ValueError("jump_fn must preserve state shape.")
        return values

    def sample_mark(
        self,
        key: Key[Array, ""],
        t: ArrayLike,
        state: ArrayLike,
        channel: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        if self.mark_fn is None:
            return jnp.zeros(self.mark_shape, dtype=jnp.asarray(state).dtype)
        values = jnp.asarray(self.mark_fn(key, t, state, channel, args))
        if values.shape != self.mark_shape:
            raise ValueError("mark_fn must return mark_shape.")
        return values


class MassActionJumpProcess(AbstractJumpProcess):
    """Well-mixed stochastic mass-action reaction system."""

    reactant_stoichiometry: Array
    product_stoichiometry: Array
    net_stoichiometry: Array
    rate_constants: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        reactant_stoichiometry: ArrayLike,
        product_stoichiometry: ArrayLike,
        rate_constants: ArrayLike,
        /,
        *,
        process_id: str | None = None,
    ):
        reactants_host = np.asarray(reactant_stoichiometry)
        products_host = np.asarray(product_stoichiometry)
        rates_host = np.asarray(rate_constants, dtype=float)
        if reactants_host.ndim != 2 or products_host.shape != reactants_host.shape:
            raise ValueError("Reactant and product stoichiometry must be equal matrices.")
        if reactants_host.shape[0] <= 0 or reactants_host.shape[1] <= 0:
            raise ValueError("Stoichiometry matrices must be non-empty.")
        if not np.all(np.isfinite(reactants_host)) or not np.all(
            np.isfinite(products_host)
        ):
            raise ValueError("Stoichiometry must be finite.")
        if np.any(reactants_host < 0) or np.any(products_host < 0):
            raise ValueError("Stoichiometry must be nonnegative.")
        if not np.all(reactants_host == np.floor(reactants_host)) or not np.all(
            products_host == np.floor(products_host)
        ):
            raise ValueError("Stoichiometry must contain integers.")
        if rates_host.shape != (reactants_host.shape[0],):
            raise ValueError("rate_constants must have one value per reaction channel.")
        if np.any(~np.isfinite(rates_host)) or np.any(rates_host < 0.0):
            raise ValueError("rate_constants must be finite and nonnegative.")
        reactants = jnp.asarray(reactants_host, dtype=jnp.int32)
        products = jnp.asarray(products_host, dtype=jnp.int32)
        rates = jnp.asarray(rates_host, dtype=float)
        resolved_id = process_id or _hash_parts(
            b"phydrax-mass-action\0",
            reactants,
            products,
            rates,
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("process_id must be a non-empty string or None.")
        self.reactant_stoichiometry = reactants
        self.product_stoichiometry = products
        self.net_stoichiometry = products - reactants
        self.rate_constants = rates
        self.state_shape = (int(reactants.shape[1]),)
        self.num_channels = int(reactants.shape[0])
        self.mark_shape = ()
        self.process_id = resolved_id

    def intensities(self, t: ArrayLike, state: ArrayLike, args: Any = None, /) -> Array:
        del t, args
        values = jnp.asarray(state, dtype=float)
        reactants = self.reactant_stoichiometry.astype(values.dtype)
        feasible = jnp.all(values[..., None, :] >= reactants, axis=-1)
        log_combinations = jnp.sum(
            jsp_special_gammaln(values[..., None, :] + 1.0)
            - jsp_special_gammaln(reactants + 1.0)
            - jsp_special_gammaln(values[..., None, :] - reactants + 1.0),
            axis=-1,
        )
        return jnp.where(
            feasible,
            self.rate_constants * jnp.exp(log_combinations),
            0.0,
        )

    def jump(
        self,
        state: ArrayLike,
        channel: ArrayLike,
        mark: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del mark, args
        return jnp.asarray(state) + self.net_stoichiometry[jnp.asarray(channel)]

    def sample_mark(
        self,
        key: Key[Array, ""],
        t: ArrayLike,
        state: ArrayLike,
        channel: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del key, t, channel, args
        return jnp.asarray(0, dtype=jnp.asarray(state).dtype)

    def conservation_residual(self, weights: ArrayLike, /) -> Array:
        values = jnp.asarray(weights, dtype=float)
        if values.shape[-1:] != self.state_shape:
            raise ValueError("weights must have a trailing species axis.")
        return jnp.einsum("...s,ks->...k", values, self.net_stoichiometry)


def jsp_special_gammaln(value: ArrayLike, /) -> Array:
    """Local indirection keeps mass-action tracing independent of SciPy objects."""
    return jax.scipy.special.gammaln(jnp.asarray(value))


__all__ = [
    "AbstractJumpProcess",
    "IntensityFunction",
    "JUMP_INVALID_INTENSITY",
    "JUMP_MAX_EVENTS",
    "JUMP_SOLVER_FAILURE",
    "JUMP_SUCCESS",
    "JumpEventBatch",
    "JumpMap",
    "JumpProcess",
    "JumpSide",
    "JumpStatus",
    "jump_status_name",
    "MarkSampler",
    "MassActionJumpProcess",
    "PoissonClockRealization",
]
