#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._interpolation import cubic_hermite_segment, linear_segment, local_cubic_slope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._irregular_trajectory_dataset import IrregularTrajectoryDatasetDomain
from ._structure import PointBatch
from ._trajectory_dataset import (
    TRAJECTORY_CASE_INDEX_KEY,
    TRAJECTORY_TIME_INDEX_KEY,
    TrajectoryDatasetDomain,
)


RaggedTimeSeriesHardInterpolation = Literal["linear", "cubic_hermite"]
RaggedTimeSeriesHardGate = Literal["sin2", "sin4"]


def _field_array(batch: PointBatch, key: str, /) -> Array:
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
        batch: PointBatch,
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
        return local_cubic_slope(
            values[case_idx, prev_idx],
            values[case_idx, node_idx],
            values[case_idx, next_idx],
            previous_width=dt,
            next_width=dt,
            has_previous=node_idx > 0,
            has_next=node_idx < lengths - 1,
        )

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
        target = linear_segment(y0, y1, s, dt)
        node_target = values[case_idx, node_idx]
        on_node_b = _broadcast_like(on_node.astype(float), target)
        target = on_node_b * node_target + (1.0 - on_node_b) * target
        if int(max_order) == 0:
            return (target,)
        target_dt = linear_segment(y0, y1, s, dt, derivative_order=1)
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

        target = cubic_hermite_segment(y0, y1, m0, m1, s, dt)
        node_target = values[case_idx, node_idx]
        on_node_b = _broadcast_like(on_node.astype(float), target)
        target = on_node_b * node_target + (1.0 - on_node_b) * target
        if int(max_order) == 0:
            return (target,)

        target_dt = cubic_hermite_segment(
            y0,
            y1,
            m0,
            m1,
            s,
            dt,
            derivative_order=1,
        )
        if int(max_order) == 1:
            return (target, target_dt)

        target_d2t = cubic_hermite_segment(
            y0,
            y1,
            m0,
            m1,
            s,
            dt,
            derivative_order=2,
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
        batch: PointBatch,
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
