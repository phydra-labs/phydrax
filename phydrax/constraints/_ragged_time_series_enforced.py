#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..domain._function import BatchAwareCallable, DomainFunction
from ..domain._irregular_trajectory_dataset import IrregularTrajectoryDatasetDomain
from ..domain._structure import CoordSeparableBatch, PointsBatch
from ..domain._trajectory_dataset import (
    TRAJECTORY_CASE_INDEX_KEY,
    TRAJECTORY_TIME_INDEX_KEY,
    TrajectoryDatasetDomain,
)
from ..operators.differential._hooks import with_derivative_hook


RaggedTimeSeriesHardInterpolation = Literal["linear", "cubic_hermite"]
RaggedTimeSeriesHardGate = Literal["sin2", "sin4"]


def _field_array(batch: PointsBatch, key: str, /) -> Array:
    if key not in batch:
        raise ValueError(
            "Ragged time-series evaluation requires trajectory batches "
            f"with internal field {key!r}."
        )
    field = batch[key]
    if not isinstance(field, cx.Field):
        raise TypeError(f"Expected batch[{key!r}] to be a coordax.Field.")
    return jnp.asarray(field.data)


def _broadcast_like(values: Array, reference: Array, /) -> Array:
    arr = jnp.asarray(values, dtype=float)
    ref = jnp.asarray(reference, dtype=float)
    if arr.ndim == ref.ndim:
        return arr
    if arr.ndim != 1:
        raise ValueError(
            f"Cannot broadcast shape {arr.shape} against reference shape {ref.shape}."
        )
    return arr.reshape((int(arr.shape[0]),) + (1,) * (ref.ndim - 1))


def _validate_values(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
) -> Array:
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim < 2:
        raise ValueError("values must have shape (N, T, ...) with a time axis.")
    if int(arr.shape[0]) != domain.size:
        raise ValueError(
            f"values leading axis must be N={domain.size}, got {arr.shape[0]}."
        )
    if int(arr.shape[1]) < domain.max_length:
        raise ValueError(
            "values time axis must have at least "
            f"{domain.max_length} entries, got {arr.shape[1]}."
        )
    return arr


def _validate_components(components: Sequence[int] | None, /) -> tuple[int, ...] | None:
    if components is None:
        return None
    out = tuple(int(component) for component in components)
    if not out:
        raise ValueError("components must be non-empty when provided.")
    if any(component < 0 for component in out):
        raise ValueError("components must contain non-negative indices.")
    if len(set(out)) != len(out):
        raise ValueError("components must not contain duplicate indices.")
    return out


def _blend_components(
    free: Array,
    hard: Array,
    components: tuple[int, ...] | None,
    /,
) -> Array:
    free_arr = jnp.asarray(free, dtype=float)
    hard_arr = jnp.asarray(hard, dtype=float)
    if components is None:
        return hard_arr
    if free_arr.ndim < 2:
        raise ValueError("components require a vector-valued trailing output axis.")
    width = int(free_arr.shape[-1])
    for component in components:
        if component >= width:
            raise ValueError(
                f"component index {component} is out of bounds for output width {width}."
            )
    component_idx = jnp.asarray(components, dtype=jnp.int32)
    mask = jnp.zeros((width,), dtype=float).at[component_idx].set(1.0)
    mask = mask.reshape((1,) * (free_arr.ndim - 1) + (width,))
    return free_arr + mask * (hard_arr - free_arr)


class _RaggedTimeSeriesTable(StrictModule, NonTrainableState):
    domain: TrajectoryDatasetDomain
    values: Array
    interpolation: RaggedTimeSeriesHardInterpolation
    gate: RaggedTimeSeriesHardGate
    snap_tol: Array

    def __init__(
        self,
        *,
        domain: TrajectoryDatasetDomain,
        values: ArrayLike,
        interpolation: RaggedTimeSeriesHardInterpolation,
        gate: RaggedTimeSeriesHardGate,
        snap_tol: float,
    ):
        if interpolation not in ("linear", "cubic_hermite"):
            raise ValueError("interpolation must be either 'linear' or 'cubic_hermite'.")
        if gate not in ("sin2", "sin4"):
            raise ValueError("gate must be either 'sin2' or 'sin4'.")
        snap = float(snap_tol)
        if snap < 0.0:
            raise ValueError("snap_tol must be non-negative.")
        self.domain = domain
        self.values = _validate_values(domain, values)
        self.interpolation = interpolation
        self.gate = gate
        self.snap_tol = jnp.asarray(snap, dtype=float)

    def max_derivative_order(self) -> int:
        if self.interpolation == "linear":
            return 1
        return 2

    def _geometry(
        self,
        batch: PointsBatch,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        case_idx = _field_array(batch, TRAJECTORY_CASE_INDEX_KEY).astype(jnp.int32)
        time_idx = _field_array(batch, TRAJECTORY_TIME_INDEX_KEY).astype(jnp.int32)
        t = _field_array(batch, self.domain.time_label).astype(float)
        values = jax.lax.stop_gradient(self.values)
        start = jax.lax.stop_gradient(self.domain.start)
        dt = jax.lax.stop_gradient(self.domain.dt)

        lengths = self.domain.lengths[case_idx]
        max_tau = lengths.astype(float) - 1.0
        tau_raw = (t - start) / dt
        tau = jnp.clip(tau_raw, 0.0, max_tau)

        max_left = jnp.maximum(lengths - 2, 0)
        k0 = jnp.floor(tau).astype(jnp.int32)
        k0 = jnp.clip(k0, 0, max_left)
        k1 = jnp.minimum(k0 + 1, lengths - 1)
        s = jnp.where(lengths > 1, tau - k0.astype(float), 0.0)

        node_idx = jnp.clip(time_idx, 0, lengths - 1)
        node_t = self.domain.observation_times(case_idx, node_idx)
        on_node = jnp.abs(t - node_t) <= self.snap_tol
        return case_idx, k0, k1, s, on_node, node_idx, lengths, values, dt

    def _node_slope(
        self,
        values: Array,
        case_idx: Array,
        node_idx: Array,
        lengths: Array,
        dt: Array,
        /,
    ) -> Array:
        prev_idx = jnp.maximum(node_idx - 1, 0)
        next_idx = jnp.minimum(node_idx + 1, lengths - 1)
        steps = jnp.maximum(next_idx - prev_idx, 1).astype(float)
        diff = values[case_idx, next_idx] - values[case_idx, prev_idx]
        return diff / (dt * _broadcast_like(steps, diff))

    def _linear_targets(
        self,
        *,
        max_order: int,
        case_idx: Array,
        k0: Array,
        k1: Array,
        s: Array,
        on_node: Array,
        node_idx: Array,
        values: Array,
        dt: Array,
    ) -> tuple[Array, ...]:
        y0 = values[case_idx, k0]
        y1 = values[case_idx, k1]
        s_b = _broadcast_like(s, y0)
        target = (1.0 - s_b) * y0 + s_b * y1
        node_target = values[case_idx, node_idx]
        on_node_b = _broadcast_like(on_node.astype(float), target)
        target = on_node_b * node_target + (1.0 - on_node_b) * target
        if int(max_order) == 0:
            return (target,)
        target_dt = (y1 - y0) / dt
        return (target, target_dt)

    def _cubic_hermite_targets(
        self,
        *,
        max_order: int,
        case_idx: Array,
        k0: Array,
        k1: Array,
        s: Array,
        on_node: Array,
        node_idx: Array,
        lengths: Array,
        values: Array,
        dt: Array,
    ) -> tuple[Array, ...]:
        y0 = values[case_idx, k0]
        y1 = values[case_idx, k1]
        m0 = self._node_slope(values, case_idx, k0, lengths, dt)
        m1 = self._node_slope(values, case_idx, k1, lengths, dt)

        s2 = s * s
        s3 = s2 * s
        h00 = _broadcast_like(2.0 * s3 - 3.0 * s2 + 1.0, y0)
        h10 = _broadcast_like(s3 - 2.0 * s2 + s, y0)
        h01 = _broadcast_like(-2.0 * s3 + 3.0 * s2, y0)
        h11 = _broadcast_like(s3 - s2, y0)
        target = h00 * y0 + h10 * dt * m0 + h01 * y1 + h11 * dt * m1

        node_target = values[case_idx, node_idx]
        on_node_b = _broadcast_like(on_node.astype(float), target)
        target = on_node_b * node_target + (1.0 - on_node_b) * target
        if int(max_order) == 0:
            return (target,)

        h00_d1 = _broadcast_like(6.0 * s2 - 6.0 * s, y0)
        h10_d1 = _broadcast_like(3.0 * s2 - 4.0 * s + 1.0, y0)
        h01_d1 = _broadcast_like(-6.0 * s2 + 6.0 * s, y0)
        h11_d1 = _broadcast_like(3.0 * s2 - 2.0 * s, y0)
        target_dt = (h00_d1 * y0 + h10_d1 * dt * m0 + h01_d1 * y1 + h11_d1 * dt * m1) / dt
        if int(max_order) == 1:
            return (target, target_dt)

        h00_d2 = _broadcast_like(12.0 * s - 6.0, y0)
        h10_d2 = _broadcast_like(6.0 * s - 4.0, y0)
        h01_d2 = _broadcast_like(-12.0 * s + 6.0, y0)
        h11_d2 = _broadcast_like(6.0 * s - 2.0, y0)
        target_d2t = (h00_d2 * y0 + h10_d2 * dt * m0 + h01_d2 * y1 + h11_d2 * dt * m1) / (
            dt * dt
        )
        return (target, target_dt, target_d2t)

    def _sin2_gate_derivatives(
        self,
        s: Array,
        on_node: Array,
        dt: Array,
        max_order: int,
        /,
    ) -> tuple[Array, ...]:
        pi_s = jnp.pi * s
        gate = jnp.sin(pi_s) ** 2
        gate = jnp.where(on_node, 0.0, gate)
        if int(max_order) == 0:
            return (gate,)
        gate_dt = (jnp.pi / dt) * jnp.sin(2.0 * pi_s)
        gate_dt = jnp.where(on_node, 0.0, gate_dt)
        if int(max_order) == 1:
            return (gate, gate_dt)
        gate_d2t = (2.0 * jnp.pi * jnp.pi / (dt * dt)) * jnp.cos(2.0 * pi_s)
        return (gate, gate_dt, gate_d2t)

    def _sin4_gate_derivatives(
        self,
        s: Array,
        on_node: Array,
        dt: Array,
        max_order: int,
        /,
    ) -> tuple[Array, ...]:
        sin_pi = jnp.sin(jnp.pi * s)
        cos_pi = jnp.cos(jnp.pi * s)
        sin2 = sin_pi * sin_pi
        gate = sin2 * sin2
        gate = jnp.where(on_node, 0.0, gate)
        if int(max_order) == 0:
            return (gate,)
        gate_dt = (4.0 * jnp.pi / dt) * sin2 * sin_pi * cos_pi
        gate_dt = jnp.where(on_node, 0.0, gate_dt)
        if int(max_order) == 1:
            return (gate, gate_dt)
        gate_d2t = (
            4.0
            * jnp.pi
            * jnp.pi
            / (dt * dt)
            * (3.0 * sin2 * cos_pi * cos_pi - sin2 * sin2)
        )
        gate_d2t = jnp.where(on_node, 0.0, gate_d2t)
        return (gate, gate_dt, gate_d2t)

    def evaluate(
        self,
        batch: PointsBatch,
        /,
        *,
        max_order: int,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        order = int(max_order)
        limit = self.max_derivative_order()
        if order > limit:
            raise ValueError(
                f"interpolation={self.interpolation!r} supports hard time "
                f"derivatives only up to order {limit}."
            )
        (
            case_idx,
            k0,
            k1,
            s,
            on_node,
            node_idx,
            lengths,
            values,
            dt,
        ) = self._geometry(batch)

        if self.interpolation == "linear":
            targets = self._linear_targets(
                max_order=order,
                case_idx=case_idx,
                k0=k0,
                k1=k1,
                s=s,
                on_node=on_node,
                node_idx=node_idx,
                values=values,
                dt=dt,
            )
        else:
            targets = self._cubic_hermite_targets(
                max_order=order,
                case_idx=case_idx,
                k0=k0,
                k1=k1,
                s=s,
                on_node=on_node,
                node_idx=node_idx,
                lengths=lengths,
                values=values,
                dt=dt,
            )

        if self.gate == "sin2":
            gates = self._sin2_gate_derivatives(s, on_node, dt, order)
        else:
            gates = self._sin4_gate_derivatives(s, on_node, dt, order)
        return targets, gates


class _RaggedTimeSeriesHardAnsatz(StrictModule, BatchAwareCallable):
    u_free: DomainFunction
    table: _RaggedTimeSeriesTable
    components: tuple[int, ...] | None

    def __init__(
        self,
        *,
        u_free: DomainFunction,
        table: _RaggedTimeSeriesTable,
        components: tuple[int, ...] | None,
    ):
        self.u_free = u_free
        self.table = table
        self.components = components

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError(
            "Ragged time-series hard enforcement requires PointsBatch evaluation."
        )

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointsBatch):
            raise TypeError(
                "Ragged time-series hard enforcement requires PointsBatch evaluation."
            )
        free = self.u_free(batch, key=key, **kwargs)
        targets, gates = self.table.evaluate(batch, max_order=0)
        free_arr = jnp.asarray(free.data, dtype=float)
        target = targets[0]
        gate_b = _broadcast_like(gates[0], free_arr)
        hard = target + gate_b * (free_arr - target)
        out = _blend_components(free_arr, hard, self.components)
        return cx.Field(out, dims=free.dims)


class _RaggedTimeSeriesHardAnsatzDerivative(StrictModule, BatchAwareCallable):
    order: int
    u_free_derivatives: tuple[DomainFunction, ...]
    table: _RaggedTimeSeriesTable
    components: tuple[int, ...] | None

    def __init__(
        self,
        *,
        order: int,
        u_free_derivatives: tuple[DomainFunction, ...],
        table: _RaggedTimeSeriesTable,
        components: tuple[int, ...] | None,
    ):
        self.order = int(order)
        self.u_free_derivatives = tuple(u_free_derivatives)
        self.table = table
        self.components = components

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError(
            "Ragged time-series hard derivative requires PointsBatch evaluation."
        )

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointsBatch):
            raise TypeError(
                "Ragged time-series hard derivative requires PointsBatch evaluation."
            )
        targets, gates = self.table.evaluate(batch, max_order=self.order)
        free_fields = tuple(
            fn(batch, key=key, **kwargs) for fn in self.u_free_derivatives
        )
        free_arrays = tuple(jnp.asarray(field.data, dtype=float) for field in free_fields)

        hard = targets[self.order]
        for gate_order in range(self.order + 1):
            free_order = self.order - gate_order
            delta = free_arrays[free_order] - targets[free_order]
            gate_b = _broadcast_like(gates[gate_order], delta)
            hard = hard + float(comb(self.order, gate_order)) * gate_b * delta

        out = _blend_components(free_arrays[self.order], hard, self.components)
        return cx.Field(out, dims=free_fields[self.order].dims)


def enforce_ragged_time_series(
    u_free: DomainFunction,
    domain: TrajectoryDatasetDomain,
    values: ArrayLike,
    /,
    *,
    interpolation: RaggedTimeSeriesHardInterpolation = "linear",
    gate: RaggedTimeSeriesHardGate = "sin2",
    time_var: str = "t",
    components: Sequence[int] | None = None,
    snap_tol: float = 1e-10,
) -> DomainFunction:
    """Return a hard ansatz that exactly matches row-wise ragged time-series data.

    The returned `DomainFunction` is evaluated on trajectory `PointsBatch` objects
    that carry the internal row/time indices emitted by `TrajectoryDatasetDomain`.
    `interpolation="linear"` supports first-order time derivatives. Use
    `interpolation="cubic_hermite"` for second-order time derivatives, preferably
    with `gate="sin4"`.

    When `components` is provided, only those trailing output components are
    hard-enforced; the remaining components are passed through from `u_free`.
    """
    if not isinstance(domain, TrajectoryDatasetDomain):
        raise TypeError("enforce_ragged_time_series requires a TrajectoryDatasetDomain.")
    if time_var != domain.time_label:
        raise ValueError(
            f"time_var must match the trajectory time label {domain.time_label!r}."
        )
    if not u_free.domain.equivalent(domain):
        raise ValueError("u_free must be defined on the provided trajectory domain.")

    components_ = _validate_components(components)
    table = _RaggedTimeSeriesTable(
        domain=domain,
        values=values,
        interpolation=interpolation,
        gate=gate,
        snap_tol=float(snap_tol),
    )

    base = DomainFunction(
        domain=u_free.domain,
        deps=u_free.deps,
        func=_RaggedTimeSeriesHardAnsatz(
            u_free=u_free,
            table=table,
            components=components_,
        ),
        metadata={},
    )

    def _make_hook(offset: int, /):
        def _hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode,
            backend,
            basis,
            periodic: bool,
        ) -> DomainFunction | None:
            if backend not in ("ad", "jet"):
                return None
            if var != domain.time_label:
                return None
            if axis is not None:
                return None
            n = int(offset) + int(order)
            return _make_derivative(
                n,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )

        return _hook

    def _make_derivative(
        order: int,
        /,
        *,
        mode,
        backend,
        basis,
        periodic: bool,
    ) -> DomainFunction:
        n = int(order)
        if n < 0:
            raise ValueError("order must be non-negative.")
        if n == 0:
            return with_derivative_hook(base, _make_hook(0))
        limit = table.max_derivative_order()
        if n > limit:
            raise ValueError(
                f"interpolation={table.interpolation!r} supports hard time "
                f"derivatives only up to order {limit}."
            )

        from ..operators.differential._domain_ops import partial_n

        u_free_derivatives = tuple(
            partial_n(
                u_free,
                var=domain.time_label,
                axis=None,
                order=k,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )
            for k in range(n + 1)
        )
        out = DomainFunction(
            domain=u_free.domain,
            deps=u_free.deps,
            func=_RaggedTimeSeriesHardAnsatzDerivative(
                order=n,
                u_free_derivatives=u_free_derivatives,
                table=table,
                components=components_,
            ),
            metadata={},
        )
        return with_derivative_hook(out, _make_hook(n))

    return with_derivative_hook(base, _make_hook(0))


__all__ = [
    "RaggedTimeSeriesHardGate",
    "RaggedTimeSeriesHardInterpolation",
    "enforce_ragged_time_series",
]
