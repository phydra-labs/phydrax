#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, cast, Protocol

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..metrix import AbstractStateGeometry
from ._delay import DelayHistory, DelayHistoryDerivative


class _ComputedDelayHistory(Protocol):
    def evaluate(self, time: Array, /, *, left: bool = True) -> Array: ...

    def derivative(self, time: Array, /, *, left: bool = True) -> Array: ...


def _is_history_buffer(value: Any, capacity: int, /) -> bool:
    return eqx.is_array(value) and value.ndim > 0 and int(value.shape[0]) == capacity


class EmptyDelayHistory(eqx.Module):
    """Shape-correct accepted-history placeholder before the first accepted step."""

    value: Array
    derivative_value: Array | None = None

    def evaluate(self, time: Array, /, *, left: bool = True) -> Array:
        del time, left
        return self.value

    def derivative(self, time: Array, /, *, left: bool = True) -> Array:
        del time, left
        if self.derivative_value is None:
            return jnp.zeros_like(self.value)
        return self.derivative_value


class DenseDelayHistory(eqx.Module):
    """Native local interpolants for accepted Diffrax steps."""

    starts: Array
    ends: Array
    infos: Any
    size: Array
    capacity: int = eqx.field(static=True)
    interpolation_cls: Any = eqx.field(static=True)

    @classmethod
    def allocate(
        cls,
        *,
        time: Array,
        dense_info_structure: Any,
        capacity: int,
        interpolation_cls: Any,
    ) -> "DenseDelayHistory":
        def allocate_leaf(value):
            if eqx.is_array(value) or isinstance(value, jax.ShapeDtypeStruct):
                return jnp.zeros((capacity,) + value.shape, dtype=value.dtype)
            return value

        return cls(
            starts=jnp.full((capacity,), jnp.inf, dtype=jnp.result_type(time)),
            ends=jnp.full((capacity,), time, dtype=jnp.result_type(time)),
            infos=jax.tree.map(allocate_leaf, dense_info_structure),
            size=jnp.asarray(0, dtype=jnp.int32),
            capacity=capacity,
            interpolation_cls=interpolation_cls,
        )

    def append(self, t0: Array, t1: Array, dense_info: Any, /) -> "DenseDelayHistory":
        index = eqx.error_if(
            self.size,
            self.size >= self.capacity,
            "Diffrax delay history exhausted its configured capacity.",
        )

        def write(buffer, value):
            if _is_history_buffer(buffer, self.capacity):
                return buffer.at[index].set(value)
            return buffer

        infos = jax.tree.map(write, self.infos, dense_info)
        return eqx.tree_at(
            lambda history: (
                history.starts,
                history.ends,
                history.infos,
                history.size,
            ),
            self,
            (
                self.starts.at[index].set(t0),
                self.ends.at[index].set(t1),
                infos,
                index + 1,
            ),
        )

    def _interpolation(self, time: Array, left: bool, /):
        """Select the local polynomial for the requested one-sided limit.

        At an ordinary knot, ``left=True`` selects the interval ending at the
        knot and ``left=False`` selects the interval beginning there. Diffrax
        jump controllers leave a two-ULP gap around known discontinuities; a
        query in such a gap selects the preceding or following polynomial,
        respectively.
        """
        last_index = jnp.maximum(self.size - 1, 0)
        latest = self.ends[last_index]
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(time), jnp.abs(latest)))
        tolerance = 100.0 * jnp.finfo(latest.dtype).eps * scale
        checked = eqx.error_if(
            time,
            time > latest + tolerance,
            "A delayed query lies beyond accepted Diffrax history.",
        )
        query = jnp.minimum(checked, latest)
        side = "left" if left else "right"
        index = jnp.searchsorted(self.starts, query, side=side) - 1
        index = jnp.clip(index, 0, last_index)
        if not left:
            in_jump_gap = query > self.ends[index]
            index = jnp.where(in_jump_gap, jnp.minimum(index + 1, last_index), index)
        query = jnp.clip(query, self.starts[index], self.ends[index])

        def gather(buffer):
            if _is_history_buffer(buffer, self.capacity):
                return buffer[index]
            return buffer

        dense_info = jax.tree.map(gather, self.infos)
        interpolation = self.interpolation_cls(
            t0=self.starts[index],
            t1=self.ends[index],
            **dense_info,
        )
        return interpolation, query

    def evaluate(self, time: Array, /, *, left: bool = True) -> Array:
        interpolation, query = self._interpolation(time, left)
        return interpolation.evaluate(query, left=left)

    def derivative(self, time: Array, /, *, left: bool = True) -> Array:
        interpolation, query = self._interpolation(time, left)
        return interpolation.derivative(query, left=left)

    def values(self, times: Array, /, *, left: bool = True) -> Array:
        flat = jnp.asarray(times).reshape((-1,))
        values = jax.vmap(lambda time: self.evaluate(time, left=left))(flat)
        return values.reshape(jnp.shape(times) + values.shape[1:])

    def derivatives(self, times: Array, /, *, left: bool = True) -> Array:
        flat = jnp.asarray(times).reshape((-1,))
        values = jax.vmap(lambda time: self.derivative(time, left=left))(flat)
        return values.reshape(jnp.shape(times) + values.shape[1:])


class RollingDelayHistory(eqx.Module):
    """Circular native interpolants for the live accepted-delay window.

    Physical storage may wrap. ``start`` and ``size`` define the logical oldest-to-
    newest order, and entries are evicted only once their end lies strictly before
    ``new_end - maximum_lag``. An append that cannot evict enough entries preserves
    the existing buffer and records ``overflowed`` for the solver driver.
    """

    starts: Array
    ends: Array
    interpolation_ends: Array
    infos: Any
    start: Array
    size: Array
    overflowed: Array
    max_size: Array
    num_evictions: Array
    overflow_time: Array
    visible_end: Array
    maximum_lag: Array
    capacity: int = eqx.field(static=True)
    interpolation_cls: Any = eqx.field(static=True)

    @classmethod
    def allocate(
        cls,
        *,
        time: Array,
        dense_info_structure: Any,
        capacity: int,
        interpolation_cls: Any,
        maximum_lag: Array,
    ) -> "RollingDelayHistory":
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("Rolling delay history capacity must be a positive integer.")

        def allocate_leaf(value):
            if eqx.is_array(value) or isinstance(value, jax.ShapeDtypeStruct):
                return jnp.zeros((capacity,) + value.shape, dtype=value.dtype)
            return value

        return cls(
            starts=jnp.full((capacity,), jnp.inf, dtype=jnp.result_type(time)),
            ends=jnp.full((capacity,), time, dtype=jnp.result_type(time)),
            interpolation_ends=jnp.full((capacity,), time, dtype=jnp.result_type(time)),
            infos=jax.tree.map(allocate_leaf, dense_info_structure),
            start=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
            max_size=jnp.asarray(0, dtype=jnp.int32),
            num_evictions=jnp.asarray(0, dtype=jnp.int32),
            overflow_time=jnp.asarray(time),
            visible_end=jnp.asarray(time),
            overflowed=jnp.asarray(False),
            maximum_lag=jnp.asarray(maximum_lag, dtype=jnp.result_type(time)),
            capacity=capacity,
            interpolation_cls=interpolation_cls,
        )

    def _physical_indices(self) -> Array:
        offsets = jnp.arange(self.capacity, dtype=jnp.int32)
        return (self.start + offsets) % self.capacity

    @property
    def logical_starts(self) -> Array:
        offsets = jnp.arange(self.capacity, dtype=jnp.int32)
        physical = self._physical_indices()
        return jnp.where(offsets < self.size, self.starts[physical], jnp.inf)

    @property
    def logical_ends(self) -> Array:
        offsets = jnp.arange(self.capacity, dtype=jnp.int32)
        physical = self._physical_indices()
        visible = jnp.minimum(self.ends[physical], self.visible_end)
        return jnp.where(offsets < self.size, visible, jnp.inf)

    @property
    def allocated_bytes(self) -> int:
        """Static bytes occupied by circular array storage."""
        leaves = (
            self.starts,
            self.ends,
            self.interpolation_ends,
            *jax.tree.leaves(self.infos),
        )
        return sum(
            int(leaf.size) * int(leaf.dtype.itemsize)
            for leaf in leaves
            if eqx.is_array(leaf)
        )

    @property
    def retained_interval(self) -> tuple[Array, Array]:
        """Closed interval available for dense evaluation."""

        oldest = jnp.take_along_axis(
            self.starts,
            self.start[..., None],
            axis=-1,
        )[..., 0]
        lower = jnp.where(self.size > 0, oldest, self.visible_end)
        return lower, self.visible_end

    def append(self, t0: Array, t1: Array, dense_info: Any, /) -> "RollingDelayHistory":
        cutoff = t1 - self.maximum_lag
        offsets = jnp.arange(self.capacity, dtype=jnp.int32)
        physical = self._physical_indices()
        interpolation_ends = self.interpolation_ends[physical]
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(cutoff), jnp.abs(interpolation_ends))
        )
        tolerance = 100.0 * jnp.finfo(interpolation_ends.dtype).eps * scale
        expired = (offsets < self.size) & (interpolation_ends < cutoff - tolerance)
        prune_count = jnp.sum(expired, dtype=jnp.int32)
        retained_start = (self.start + prune_count) % self.capacity
        retained_size = self.size - prune_count
        can_append = retained_size < self.capacity
        write_index = (retained_start + retained_size) % self.capacity

        def write(buffer, value):
            if not _is_history_buffer(buffer, self.capacity):
                return buffer
            return jax.lax.cond(
                can_append,
                lambda current: current.at[write_index].set(value),
                lambda current: current,
                buffer,
            )

        infos = jax.tree.map(write, self.infos, dense_info)
        starts = write(self.starts, t0)
        ends = write(self.ends, t1)
        interpolation_ends = write(self.interpolation_ends, t1)
        next_size = retained_size + can_append.astype(jnp.int32)
        return eqx.tree_at(
            lambda history: (
                history.starts,
                history.ends,
                history.interpolation_ends,
                history.infos,
                history.start,
                history.size,
                history.max_size,
                history.num_evictions,
                history.overflowed,
                history.overflow_time,
                history.visible_end,
            ),
            self,
            (
                starts,
                ends,
                interpolation_ends,
                infos,
                retained_start,
                next_size,
                jnp.maximum(self.max_size, next_size),
                self.num_evictions + prune_count,
                self.overflowed | ~can_append,
                jnp.where(can_append, self.overflow_time, t1),
                jnp.where(can_append, t1, self.visible_end),
            ),
        )

    def with_visible_end(self, final_time: Array, /) -> "RollingDelayHistory":
        """Hide an accepted event-crossing tail without rescaling its interpolant."""
        visible = jnp.minimum(self.visible_end, jnp.asarray(final_time))
        return eqx.tree_at(
            lambda history: (history.ends, history.visible_end),
            self,
            (jnp.minimum(self.ends, visible), visible),
        )

    def latest_info(self) -> Any:
        index = (self.start + jnp.maximum(self.size - 1, 0)) % self.capacity

        def gather(buffer):
            return buffer[index] if _is_history_buffer(buffer, self.capacity) else buffer

        return jax.tree.map(gather, self.infos)

    def _interpolation(self, time: Array, left: bool, /):
        physical = self._physical_indices()
        logical_starts = self.logical_starts
        last_offset = jnp.maximum(self.size - 1, 0)
        latest = jnp.minimum(self.ends[physical[last_offset]], self.visible_end)
        earliest = jnp.where(self.size > 0, logical_starts[0], self.visible_end)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(time), jnp.maximum(jnp.abs(earliest), jnp.abs(latest))),
        )
        tolerance = 100.0 * jnp.finfo(latest.dtype).eps * scale
        checked = eqx.error_if(
            time,
            (time < earliest - tolerance) | (time > latest + tolerance),
            "A delayed query lies outside retained accepted history.",
        )
        query = jnp.clip(checked, earliest, latest)
        side = "left" if left else "right"
        logical_index = jnp.searchsorted(logical_starts, query, side=side) - 1
        logical_index = jnp.clip(logical_index, 0, last_offset)
        if not left:
            in_jump_gap = query > self.ends[physical[logical_index]]
            logical_index = jnp.where(
                in_jump_gap, jnp.minimum(logical_index + 1, last_offset), logical_index
            )
        index = physical[logical_index]
        query = jnp.clip(query, self.starts[index], self.ends[index])

        def gather(buffer):
            return buffer[index] if _is_history_buffer(buffer, self.capacity) else buffer

        dense_info = jax.tree.map(gather, self.infos)
        interpolation = self.interpolation_cls(
            t0=self.starts[index],
            t1=self.interpolation_ends[index],
            **dense_info,
        )
        return interpolation, query

    def evaluate(self, time: Array, /, *, left: bool = True) -> Array:
        interpolation, query = self._interpolation(time, left)
        return interpolation.evaluate(query, left=left)

    def derivative(self, time: Array, /, *, left: bool = True) -> Array:
        interpolation, query = self._interpolation(time, left)
        return interpolation.derivative(query, left=left)

    def values(self, times: Array, /, *, left: bool = True) -> Array:
        query = jnp.asarray(times)
        flat = query.reshape((-1,))
        values = jax.vmap(lambda time: self.evaluate(time, left=left))(flat)
        return values.reshape(query.shape + values.shape[1:])

    def derivatives(self, times: Array, /, *, left: bool = True) -> Array:
        query = jnp.asarray(times)
        flat = query.reshape((-1,))
        values = jax.vmap(lambda time: self.derivative(time, left=left))(flat)
        return values.reshape(query.shape + values.shape[1:])


class DelayHistoryView(eqx.Module):
    """One view combining prehistory with accepted local interpolants."""

    initial_history: DelayHistory
    initial_derivative: DelayHistoryDerivative | None
    args: Any
    initial_time: Array
    computed_history: _ComputedDelayHistory
    state_shape: tuple[int, ...] = eqx.field(static=True)
    geometry: AbstractStateGeometry | None
    derivative_shape: tuple[int, ...] | None = eqx.field(
        static=True,
        default=None,
    )

    def value(self, time: Array, /, *, left: bool = True) -> Array:
        def from_initial(query):
            value = jnp.asarray(self.initial_history(query, self.args))
            if value.shape != self.state_shape:
                raise ValueError("Delay history changed its declared state shape.")
            return value

        def from_computed(query):
            return self.computed_history.evaluate(query, left=left)

        value = jax.lax.cond(
            time <= self.initial_time,
            from_initial,
            from_computed,
            time,
        )
        if self.geometry is not None:
            membership = jnp.asarray(self.geometry.contains(value), dtype=bool)
            if membership.shape != ():
                raise ValueError(
                    "State geometry contains() must return a scalar boolean."
                )
            value = eqx.error_if(
                value,
                ~membership,
                "A delayed history value lies outside state_geometry.",
            )
        return value

    def derivative(self, time: Array, /, *, left: bool = True) -> Array:
        use_initial = (time < self.initial_time) | ((time == self.initial_time) & left)
        if self.initial_derivative is None:
            value = self.computed_history.derivative(time, left=left)
            return eqx.error_if(
                value,
                use_initial,
                "Delay history does not define a derivative callback.",
            )
        initial_derivative = self.initial_derivative

        def from_initial(query):
            value = jnp.asarray(initial_derivative(query, self.args))
            expected = (
                self.state_shape
                if self.derivative_shape is None
                else self.derivative_shape
            )
            if value.shape != expected:
                raise ValueError(
                    "Delay history derivative changed its declared tangent shape."
                )
            return value

        def from_computed(query):
            return self.computed_history.derivative(query, left=left)

        use_initial = (time < self.initial_time) | ((time == self.initial_time) & left)
        return jax.lax.cond(use_initial, from_initial, from_computed, time)

    def values(self, times: ArrayLike, /, *, left: bool = True) -> Array:
        query = jnp.asarray(times)
        flat = query.reshape((-1,))
        values = jax.lax.map(lambda time: self.value(time, left=left), flat)
        return values.reshape(query.shape + values.shape[1:])

    def derivatives(self, times: ArrayLike, /, *, left: bool = True) -> Array:
        query = jnp.asarray(times)
        flat = query.reshape((-1,))
        values = jax.lax.map(lambda time: self.derivative(time, left=left), flat)
        return values.reshape(query.shape + values.shape[1:])


class DelayDenseInterpolation(eqx.Module):
    """Dense values from a solved delay interval."""

    history: DelayHistoryView
    final_time: Array
    lower_time: Array | None = None

    @property
    def computed_history(self) -> DenseDelayHistory | RollingDelayHistory:
        """Accepted local interpolants backing this dense solution."""
        return cast(
            DenseDelayHistory | RollingDelayHistory, self.history.computed_history
        )

    @eqx.filter_jit
    def evaluate(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = jnp.asarray(query_times)
        if jnp.iscomplexobj(query):
            raise TypeError("Dense delay query times must be real-valued.")
        if query.size == 0:
            raise ValueError("Dense delay query times must be non-empty.")
        query = query.astype(float)
        query = eqx.error_if(
            query,
            ~jnp.all(jnp.isfinite(query)),
            "Dense delay query times must be finite.",
        )
        lower = (
            self.history.initial_time
            if self.lower_time is None
            else jnp.asarray(self.lower_time)
        )
        query = eqx.error_if(
            query,
            jnp.any((query < lower) | (query > self.final_time)),
            "Dense delay query times must lie within the solved interval "
            "(the available solved interval for rolling history).",
        )
        return self.history.values(query, left=left)

    @eqx.filter_jit
    def derivative(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        """Evaluate one-sided derivatives of the accepted dense solution."""
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = jnp.asarray(query_times)
        if jnp.iscomplexobj(query):
            raise TypeError("Dense delay query times must be real-valued.")
        if query.size == 0:
            raise ValueError("Dense delay query times must be non-empty.")
        query = query.astype(float)
        query = eqx.error_if(
            query,
            ~jnp.all(jnp.isfinite(query)),
            "Dense delay query times must be finite.",
        )
        lower = (
            self.history.initial_time
            if self.lower_time is None
            else jnp.asarray(self.lower_time)
        )
        query = eqx.error_if(
            query,
            jnp.any((query < lower) | (query > self.final_time)),
            "Dense delay query times must lie within the solved interval "
            "(the available solved interval for rolling history).",
        )
        return self.history.derivatives(query, left=left)


def interpolation_derivative_supported(interpolation_cls: Any, /) -> bool:
    """Whether an interpolation class implements the Diffrax local interface."""
    return isinstance(interpolation_cls, type) and issubclass(
        interpolation_cls, dfx.AbstractLocalInterpolation
    )


__all__ = [
    "DelayDenseInterpolation",
    "DelayHistoryView",
    "DenseDelayHistory",
    "RollingDelayHistory",
    "EmptyDelayHistory",
    "interpolation_derivative_supported",
]
