#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._interpolation import (
    apply_gather_stencil,
    linear_stencil_from_indices,
    nearest_stencil_from_indices,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._derivative import CallbackDerivativeRule
from ._evaluation import BatchEvaluator
from ._function import DomainFunction
from ._irregular_trajectory_dataset import IrregularTrajectoryDatasetDomain
from ._structure import GridBatch, PointBatch
from ._trajectory_dataset import TRAJECTORY_CASE_INDEX_KEY, TrajectoryDatasetDomain
from ._trajectory_interpolation import _RaggedTimeSeriesTable, _validate_values


TrajectorySignalInterpolation = Literal["nearest", "linear", "cubic_hermite"]

class _NearestTrajectorySignal(StrictModule, BatchEvaluator, NonTrainableState):
    domain: TrajectoryDatasetDomain
    values: Array

    def __init__(self, *, domain: TrajectoryDatasetDomain, values: ArrayLike):
        self.domain = domain
        self.values = _validate_values(domain, values)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError("TrajectorySignal requires PointBatch evaluation.")
        if TRAJECTORY_CASE_INDEX_KEY not in batch:
            raise ValueError(
                "TrajectorySignal requires trajectory batches with internal case indices."
            )
        case_field = batch[TRAJECTORY_CASE_INDEX_KEY]
        time_field = batch[self.domain.time_label]
        if not isinstance(case_field, cx.Field):
            raise TypeError("Trajectory case indices must be stored as a Field.")
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        case_idx = jnp.asarray(case_field.data, dtype=jnp.int32)
        t = jnp.asarray(time_field.data, dtype=float)
        tau = jnp.rint((t - self.domain.start) / self.domain.dt).astype(jnp.int32)
        lengths = self.domain.lengths[case_idx]
        time_idx = jnp.clip(tau, 0, lengths - 1)
        values = jax.lax.stop_gradient(self.values)
        time_count = int(values.shape[1])
        source = values.reshape((-1,) + values.shape[2:])
        stencil = nearest_stencil_from_indices(
            case_idx * time_count + time_idx,
            source_size=int(source.shape[0]),
        )
        out = apply_gather_stencil(source, stencil).values
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


class _IrregularNearestTrajectorySignal(StrictModule, BatchEvaluator, NonTrainableState):
    domain: IrregularTrajectoryDatasetDomain
    values: Array

    def __init__(self, *, domain: IrregularTrajectoryDatasetDomain, values: ArrayLike):
        self.domain = domain
        self.values = _validate_values(domain, values)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError("TrajectorySignal requires PointBatch evaluation.")
        if TRAJECTORY_CASE_INDEX_KEY not in batch:
            raise ValueError(
                "TrajectorySignal requires trajectory batches with internal case indices."
            )
        case_field = batch[TRAJECTORY_CASE_INDEX_KEY]
        time_field = batch[self.domain.time_label]
        if not isinstance(case_field, cx.Field):
            raise TypeError("Trajectory case indices must be stored as a Field.")
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        case_idx = jnp.asarray(case_field.data, dtype=jnp.int32)
        t = jnp.asarray(time_field.data, dtype=float)
        time_idx = self.domain.nearest_time_indices(case_idx, t)
        values = jax.lax.stop_gradient(self.values)
        time_count = int(values.shape[1])
        source = values.reshape((-1,) + values.shape[2:])
        stencil = nearest_stencil_from_indices(
            case_idx * time_count + time_idx,
            source_size=int(source.shape[0]),
        )
        out = apply_gather_stencil(source, stencil).values
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


class _InterpolatedTrajectorySignal(StrictModule, BatchEvaluator, NonTrainableState):
    table: _RaggedTimeSeriesTable
    order: int

    def __init__(self, *, table: _RaggedTimeSeriesTable, order: int = 0):
        self.table = table
        self.order = int(order)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError("TrajectorySignal requires PointBatch evaluation.")
        time_field = batch[self.table.domain.time_label]
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        targets, _gates = self.table.evaluate(batch, max_order=self.order)
        out = targets[self.order]
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


class _IrregularTrajectorySignalTable(StrictModule, NonTrainableState):
    domain: IrregularTrajectoryDatasetDomain
    values: Array

    def __init__(
        self,
        *,
        domain: IrregularTrajectoryDatasetDomain,
        values: ArrayLike,
    ):
        self.domain = domain
        self.values = _validate_values(domain, values)

    def evaluate(
        self,
        batch: PointBatch,
        /,
        *,
        max_order: int,
    ) -> tuple[Array, ...]:
        order = int(max_order)
        if order > 1:
            raise ValueError(
                "Irregular TrajectorySignal with interpolation='linear' supports "
                "time derivatives only up to order 1."
            )
        if TRAJECTORY_CASE_INDEX_KEY not in batch:
            raise ValueError(
                "TrajectorySignal requires trajectory batches with internal case indices."
            )
        case_field = batch[TRAJECTORY_CASE_INDEX_KEY]
        time_field = batch[self.domain.time_label]
        if not isinstance(case_field, cx.Field):
            raise TypeError("Trajectory case indices must be stored as a Field.")
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        case_idx = jnp.asarray(case_field.data, dtype=jnp.int32)
        times = jnp.asarray(time_field.data, dtype=float)
        lower, upper, fraction = self.domain.bracketing_time_indices(
            case_idx,
            times,
        )
        values = jax.lax.stop_gradient(self.values)
        time_count = int(values.shape[1])
        source = values.reshape((-1,) + values.shape[2:])
        lower_global = case_idx * time_count + lower
        upper_global = case_idx * time_count + upper
        target_stencil = linear_stencil_from_indices(
            lower_global,
            upper_global,
            fraction,
            source_size=int(source.shape[0]),
        )
        target = apply_gather_stencil(source, target_stencil).values
        if order == 0:
            return (target,)

        t0 = self.domain.times[case_idx, lower]
        t1 = self.domain.times[case_idx, upper]
        lengths = self.domain.lengths[case_idx]
        width = jnp.where(lengths > 1, t1 - t0, 1.0)
        derivative_stencil = linear_stencil_from_indices(
            lower_global,
            upper_global,
            fraction,
            source_size=int(source.shape[0]),
            derivative_order=1,
            interval_width=width,
        )
        derivative = apply_gather_stencil(source, derivative_stencil).values
        derivative = jnp.where(
            (lengths > 1).reshape(
                lengths.shape + (1,) * (derivative.ndim - lengths.ndim)
            ),
            derivative,
            0.0,
        )
        return (target, derivative)


class _IrregularInterpolatedTrajectorySignal(
    StrictModule, BatchEvaluator, NonTrainableState
):
    table: _IrregularTrajectorySignalTable
    order: int

    def __init__(self, *, table: _IrregularTrajectorySignalTable, order: int = 0):
        self.table = table
        self.order = int(order)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError("TrajectorySignal requires PointBatch evaluation.")
        time_field = batch[self.table.domain.time_label]
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        targets = self.table.evaluate(batch, max_order=self.order)
        out = targets[self.order]
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


def _irregular_trajectory_signal(
    domain: IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
    *,
    interpolation: TrajectorySignalInterpolation,
) -> DomainFunction:
    if interpolation == "cubic_hermite":
        raise ValueError(
            "IrregularTrajectoryDatasetDomain TrajectorySignal supports only "
            "interpolation='nearest' or interpolation='linear'."
        )

    if interpolation == "nearest":
        base = DomainFunction(
            domain=domain,
            deps=domain.labels,
            func=_IrregularNearestTrajectorySignal(domain=domain, values=values),
            metadata={},
        )

        def _nearest_hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode,
            backend,
            basis,
            periodic: bool,
        ) -> DomainFunction | None:
            del mode, backend, basis, periodic
            if var != domain.time_label or axis is not None:
                return None
            if int(order) == 0:
                return base.with_derivative_rule(CallbackDerivativeRule(_nearest_hook))
            raise ValueError(
                "TrajectorySignal with interpolation='nearest' is not differentiable; "
                "use interpolation='linear' for time derivatives."
            )

        return base.with_derivative_rule(CallbackDerivativeRule(_nearest_hook))

    table = _IrregularTrajectorySignalTable(domain=domain, values=values)
    base = DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_IrregularInterpolatedTrajectorySignal(table=table, order=0),
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
            del mode, basis, periodic
            if backend not in ("ad", "jet"):
                return None
            if var != domain.time_label:
                return None
            if axis is not None:
                return None
            n = int(offset) + int(order)
            if n > 1:
                raise ValueError(
                    "Irregular TrajectorySignal with interpolation='linear' supports "
                    "time derivatives only up to order 1."
                )
            if n == 0:
                return base.with_derivative_rule(CallbackDerivativeRule(_make_hook(0)))
            out = DomainFunction(
                domain=domain,
                deps=domain.labels,
                func=_IrregularInterpolatedTrajectorySignal(table=table, order=n),
                metadata={},
            )
            return out.with_derivative_rule(CallbackDerivativeRule(_make_hook(n)))

        return _hook

    return base.with_derivative_rule(CallbackDerivativeRule(_make_hook(0)))


def TrajectorySignal(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
    *,
    interpolation: TrajectorySignalInterpolation = "linear",
    time_var: str | None = None,
    snap_tol: float = 1e-10,
) -> DomainFunction:
    """Expose fixed trajectory data as a `DomainFunction` over `(data, t)`.

    Use this when an observed ragged time series is an input or forcing term for
    another residual, rather than the supervised output being fitted directly.
    `values` must have one leading row per trajectory case and a padded time axis
    matching the domain lengths. Interpolated signals are non-trainable solver
    state and support time derivatives according to the interpolation order.
    """
    if not isinstance(
        domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
    ):
        raise TypeError("TrajectorySignal requires a trajectory dataset domain.")
    var = domain.time_label if time_var is None else str(time_var)
    if var != domain.time_label:
        raise ValueError(
            f"time_var must match the trajectory time label {domain.time_label!r}."
        )

    interpolation_str = str(interpolation)
    if interpolation_str not in ("nearest", "linear", "cubic_hermite"):
        raise ValueError("interpolation must be 'nearest', 'linear', or 'cubic_hermite'.")

    if isinstance(domain, IrregularTrajectoryDatasetDomain):
        interpolation_value = (
            "nearest"
            if interpolation_str == "nearest"
            else ("linear" if interpolation_str == "linear" else "cubic_hermite")
        )
        return _irregular_trajectory_signal(
            domain,
            values,
            interpolation=interpolation_value,
        )

    if interpolation_str == "nearest":
        base = DomainFunction(
            domain=domain,
            deps=domain.labels,
            func=_NearestTrajectorySignal(domain=domain, values=values),
            metadata={},
        )

        def _nearest_hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode,
            backend,
            basis,
            periodic: bool,
        ) -> DomainFunction | None:
            del mode, backend, basis, periodic
            if var != domain.time_label or axis is not None:
                return None
            if int(order) == 0:
                return base.with_derivative_rule(CallbackDerivativeRule(_nearest_hook))
            raise ValueError(
                "TrajectorySignal with interpolation='nearest' is not differentiable; "
                "use interpolation='linear' or 'cubic_hermite' for time derivatives."
            )

        return base.with_derivative_rule(CallbackDerivativeRule(_nearest_hook))

    interpolation_value: Literal["linear", "cubic_hermite"]
    if interpolation_str == "linear":
        interpolation_value = "linear"
    else:
        interpolation_value = "cubic_hermite"
    table = _RaggedTimeSeriesTable(
        domain=domain,
        values=values,
        interpolation=interpolation_value,
        gate="sin2",
        snap_tol=float(snap_tol),
    )
    base = DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_InterpolatedTrajectorySignal(table=table, order=0),
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
            del mode, basis, periodic
            if backend not in ("ad", "jet"):
                return None
            if var != domain.time_label:
                return None
            if axis is not None:
                return None
            n = int(offset) + int(order)
            limit = table.max_derivative_order()
            if n > limit:
                raise ValueError(
                    f"interpolation={table.interpolation!r} supports trajectory "
                    f"signal time derivatives only up to order {limit}."
                )
            if n == 0:
                return base.with_derivative_rule(CallbackDerivativeRule(_make_hook(0)))
            out = DomainFunction(
                domain=domain,
                deps=domain.labels,
                func=_InterpolatedTrajectorySignal(table=table, order=n),
                metadata={},
            )
            return out.with_derivative_rule(CallbackDerivativeRule(_make_hook(n)))

        return _hook

    return base.with_derivative_rule(CallbackDerivativeRule(_make_hook(0)))



__all__ = ["TrajectorySignalInterpolation"]
