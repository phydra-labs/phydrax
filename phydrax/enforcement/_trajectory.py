#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    BatchEvaluator,
    CallbackDerivativeRule,
    DomainFunction,
    GridBatch,
    PointBatch,
    TrajectoryDatasetDomain,
)

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ..domain._trajectory_interpolation import (
    _broadcast_like,
    _RaggedTimeSeriesTable,
)
from ..operators.differential._hooks import with_derivative_rule


RaggedTimeSeriesHardInterpolation = Literal["linear", "cubic_hermite"]
RaggedTimeSeriesHardGate = Literal["sin2", "sin4"]


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


class _RaggedTimeSeriesHardAnsatz(StrictModule, BatchEvaluator):
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
            "Ragged time-series hard enforcement requires PointBatch evaluation."
        )

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointBatch):
            raise TypeError(
                "Ragged time-series hard enforcement requires PointBatch evaluation."
            )
        free = self.u_free(batch, key=key, **kwargs)
        targets, gates = self.table.evaluate(batch, max_order=0)
        free_arr = jnp.asarray(free.data, dtype=float)
        target = targets[0]
        gate_b = _broadcast_like(gates[0], free_arr)
        hard = target + gate_b * (free_arr - target)
        out = _blend_components(free_arr, hard, self.components)
        return cx.Field(out, dims=free.dims)


class _RaggedTimeSeriesHardAnsatzDerivative(StrictModule, BatchEvaluator):
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
            "Ragged time-series hard derivative requires PointBatch evaluation."
        )

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointBatch):
            raise TypeError(
                "Ragged time-series hard derivative requires PointBatch evaluation."
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

    The returned `DomainFunction` is evaluated on trajectory `PointBatch` objects
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
    if not u_free.domain.same_support(domain):
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
            return with_derivative_rule(base, CallbackDerivativeRule(_make_hook(0)))
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
        return with_derivative_rule(out, CallbackDerivativeRule(_make_hook(n)))

    return with_derivative_rule(base, CallbackDerivativeRule(_make_hook(0)))


__all__ = [
    "RaggedTimeSeriesHardGate",
    "RaggedTimeSeriesHardInterpolation",
    "enforce_ragged_time_series",
]
