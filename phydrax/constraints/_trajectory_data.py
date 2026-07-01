#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..domain._components import DomainComponent
from ..domain._function import BatchAwareCallable, DomainFunction
from ..domain._irregular_trajectory_dataset import IrregularTrajectoryDatasetDomain
from ..domain._structure import CoordSeparableBatch, PointsBatch, ProductStructure
from ..domain._trajectory_dataset import (
    TRAJECTORY_CASE_INDEX_KEY,
    TrajectoryDatasetDomain,
)
from ..operators.differential._hooks import with_derivative_hook
from ._base import AbstractSamplingConstraint
from ._data_metrics import (
    reduce_supervised_loss,
    sample_case_indices as _sample_case_indices_uniform,
    supervised_data_metrics,
    supervised_per_sample_squared_error,
    validate_case_indices,
    validate_supervised_targets,
)
from ._ragged_time_series_enforced import _RaggedTimeSeriesTable, _validate_values


TrajectorySignalInterpolation = Literal["nearest", "linear", "cubic_hermite"]
TrajectoryCaseTime = Literal["start", "end"] | float


class TrajectoryCaseDataBatch(StrictModule):
    """A sampled mini-batch of case-level trajectory-domain data."""

    points: PointsBatch
    target: Array
    case_indices: Array
    times: Array

    def __init__(
        self,
        *,
        points: PointsBatch,
        target: ArrayLike,
        case_indices: ArrayLike,
        times: ArrayLike,
    ):
        self.points = points
        self.target = jnp.asarray(target, dtype=float)
        self.case_indices = jnp.asarray(case_indices, dtype=jnp.int32)
        self.times = jnp.asarray(times, dtype=float)


def _validate_case_values(domain: TrajectoryDatasetDomain, values: ArrayLike, /) -> Array:
    return validate_supervised_targets(
        values,
        leading_size=domain.size,
        name="case target",
    )


def _time_selection(
    domain: TrajectoryDatasetDomain,
    case_indices: Array,
    time: TrajectoryCaseTime,
    /,
) -> tuple[Array, Array]:
    if time == "start":
        n = int(case_indices.shape[0])
        return (
            jnp.full((n,), domain.start, dtype=float),
            jnp.zeros((n,), dtype=jnp.int32),
        )
    if time == "end":
        return domain.end_times[case_indices], domain.lengths[case_indices] - 1

    value = _fixed_case_time_value(time)
    time_indices = jnp.rint((value - domain.start) / domain.dt).astype(jnp.int32)
    n = int(case_indices.shape[0])
    return (
        jnp.full((n,), value, dtype=float),
        jnp.full((n,), time_indices, dtype=jnp.int32),
    )


def _sample_case_indices(
    domain: TrajectoryDatasetDomain,
    n: int,
    key: Key[Array, ""],
    time: TrajectoryCaseTime,
    /,
    *,
    indices: Array | None = None,
) -> Array:
    if time == "start" or time == "end":
        return _sample_case_indices_uniform(
            size=domain.size,
            num_samples=n,
            key=key,
            indices=indices,
        )

    value = _fixed_case_time_value(time)
    valid = (domain.start <= value) & (value <= domain.end_times)
    if indices is not None:
        allowed = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        valid = valid[allowed]
        probs = valid.astype(float) / jnp.sum(valid.astype(float))
        positions = jr.choice(key, int(allowed.shape[0]), shape=(n,), p=probs)
        return allowed[positions].astype(jnp.int32)
    probs = valid.astype(float) / jnp.sum(valid.astype(float))
    return jr.choice(key, domain.size, shape=(n,), p=probs).astype(jnp.int32)


def _fixed_case_time_value(time: TrajectoryCaseTime, /) -> Array:
    if isinstance(time, str):
        raise ValueError("case_time must be 'start', 'end', or a floating time value.")
    return jnp.asarray(float(time), dtype=float).reshape(())


class TrajectoryCaseDataConstraint(AbstractSamplingConstraint):
    """Supervise per-case targets on a TrajectoryDatasetDomain.

    The target has one row per trajectory case, for example scalar material
    parameters, class logits, or final scalar outputs attached to each dataset row.
    Case-only functions such as `theta(data)` ignore the representative time carried
    by the sampled trajectory batch; trajectory functions `theta(data, t)` are
    evaluated at the configured `case_time`.
    """

    constraint_vars: tuple[str, ...]
    component: DomainComponent
    structure: ProductStructure
    dense_structure: ProductStructure | None
    num_points: int
    sampler: str
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "sum"]
    values: Array
    case_time: TrajectoryCaseTime
    weight: Array
    pointwise_weight: DomainFunction | None
    case_indices: Array | None
    label: str | None
    data_accuracy_eps: Array

    def __init__(
        self,
        constraint_var: str,
        component: DomainComponent,
        values: ArrayLike,
        /,
        *,
        num_cases: int,
        structure: ProductStructure | None = None,
        case_time: TrajectoryCaseTime = "start",
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        case_indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        if not isinstance(component.domain, TrajectoryDatasetDomain):
            raise TypeError(
                "TrajectoryCaseDataConstraint requires a TrajectoryDatasetDomain component."
            )
        n = int(num_cases)
        if n <= 0:
            raise ValueError("num_cases must be positive.")
        reduction_str = str(reduction)
        if reduction_str not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        reduction_value: Literal["mean", "sum"]
        if reduction_str == "mean":
            reduction_value = "mean"
        else:
            reduction_value = "sum"

        domain = component.domain
        case_indices_arr = validate_case_indices(
            case_indices,
            size=domain.size,
            name="case_indices",
        )

        if case_time != "start" and case_time != "end":
            value = _fixed_case_time_value(case_time)
            valid = (domain.start <= value) & (value <= domain.end_times)
            if case_indices_arr is not None:
                valid = valid[case_indices_arr]
            if not bool(jnp.any(valid)):
                raise ValueError(
                    "No trajectory cases are valid at the requested case time."
                )

        self.constraint_vars = (str(constraint_var),)
        self.component = component
        self.structure = structure or ProductStructure((domain.labels,))
        self.dense_structure = None
        self.num_points = n
        self.sampler = "case_uniform"
        self.over = None
        self.reduction = reduction_value
        self.values = _validate_case_values(domain, values)
        self.case_time = case_time
        if isinstance(weight, DomainFunction):
            self.weight = jnp.asarray(1.0, dtype=float)
            self.pointwise_weight = weight
        else:
            self.weight = jnp.asarray(weight, dtype=float)
            self.pointwise_weight = None
        self.case_indices = case_indices_arr
        self.label = None if label is None else str(label)
        self.data_accuracy_eps = jnp.asarray(float(data_accuracy_eps), dtype=float)

    @property
    def domain(self) -> TrajectoryDatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, TrajectoryDatasetDomain):
            raise TypeError(
                "TrajectoryCaseDataConstraint domain is not a trajectory domain."
            )
        return domain

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        domain = self.domain
        case_indices = _sample_case_indices(
            domain,
            int(self.num_points),
            key,
            self.case_time,
            indices=self.case_indices,
        )
        times, time_indices = _time_selection(domain, case_indices, self.case_time)
        points = domain.points_from_case_time(
            case_indices,
            times,
            structure=self.structure,
            time_indices=time_indices,
        )
        return TrajectoryCaseDataBatch(
            points=points,
            target=self.values[case_indices],
            case_indices=case_indices,
            times=times,
        )

    def _prediction(
        self,
        functions: Mapping[str, DomainFunction],
        batch: TrajectoryCaseDataBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        var = self.constraint_vars[0]
        prediction = functions[var](batch.points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError("Expected case data prediction to return a coordax.Field.")
        return prediction

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: TrajectoryCaseDataBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        batch_ = self.sample(key=key) if batch is None else batch
        prediction = self._prediction(functions, batch_, key=key, **kwargs)
        return supervised_data_metrics(
            jnp.asarray(prediction.data, dtype=float),
            batch_.target,
            eps=self.data_accuracy_eps,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        batch: TrajectoryCaseDataBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        batch_ = self.sample(key=key) if batch is None else batch
        prediction = self._prediction(functions, batch_, key=key, **kwargs)
        per_sample = supervised_per_sample_squared_error(
            jnp.asarray(prediction.data, dtype=float),
            batch_.target,
        )

        if self.pointwise_weight is not None:
            w = self.pointwise_weight(batch_.points, key=key, **kwargs)
            if not isinstance(w, cx.Field):
                raise TypeError("pointwise weight must return a coordax.Field.")
            w_arr = jnp.asarray(w.data, dtype=float)
            if w_arr.ndim == 0:
                per_sample = per_sample * w_arr
            else:
                per_sample = per_sample * jnp.squeeze(w_arr).reshape((-1,))

        reduced = reduce_supervised_loss(per_sample, reduction=self.reduction)
        return self.weight * jnp.asarray(reduced, dtype=float).reshape(())


class _NearestTrajectorySignal(StrictModule, BatchAwareCallable, NonTrainableState):
    domain: TrajectoryDatasetDomain
    values: Array

    def __init__(self, *, domain: TrajectoryDatasetDomain, values: ArrayLike):
        self.domain = domain
        self.values = _validate_values(domain, values)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointsBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointsBatch):
            raise TypeError("TrajectorySignal requires PointsBatch evaluation.")
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
        out = values[case_idx, time_idx]
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


class _IrregularNearestTrajectorySignal(
    StrictModule, BatchAwareCallable, NonTrainableState
):
    domain: IrregularTrajectoryDatasetDomain
    values: Array

    def __init__(
        self, *, domain: IrregularTrajectoryDatasetDomain, values: ArrayLike
    ):
        self.domain = domain
        self.values = _validate_values(domain, values)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointsBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointsBatch):
            raise TypeError("TrajectorySignal requires PointsBatch evaluation.")
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
        out = values[case_idx, time_idx]
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


class _InterpolatedTrajectorySignal(StrictModule, BatchAwareCallable, NonTrainableState):
    table: _RaggedTimeSeriesTable
    order: int

    def __init__(self, *, table: _RaggedTimeSeriesTable, order: int = 0):
        self.table = table
        self.order = int(order)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointsBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointsBatch):
            raise TypeError("TrajectorySignal requires PointsBatch evaluation.")
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
        batch: PointsBatch,
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
        t = jnp.asarray(time_field.data, dtype=float)
        lower, upper, frac = self.domain.bracketing_time_indices(case_idx, t)
        values = jax.lax.stop_gradient(self.values)
        y0 = values[case_idx, lower]
        y1 = values[case_idx, upper]
        frac_b = _broadcast_like(frac, y0)
        target = (1.0 - frac_b) * y0 + frac_b * y1
        if order == 0:
            return (target,)

        t0 = self.domain.times[case_idx, lower]
        t1 = self.domain.times[case_idx, upper]
        lengths = self.domain.lengths[case_idx]
        denom = jnp.where(lengths > 1, t1 - t0, 1.0)
        slope = (y1 - y0) / _broadcast_like(denom, y0)
        slope = jnp.where(_broadcast_like(lengths > 1, slope), slope, 0.0)
        return (target, slope)


class _IrregularInterpolatedTrajectorySignal(
    StrictModule, BatchAwareCallable, NonTrainableState
):
    table: _IrregularTrajectorySignalTable
    order: int

    def __init__(self, *, table: _IrregularTrajectorySignalTable, order: int = 0):
        self.table = table
        self.order = int(order)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointsBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointsBatch):
            raise TypeError("TrajectorySignal requires PointsBatch evaluation.")
        time_field = batch[self.table.domain.time_label]
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        targets = self.table.evaluate(batch, max_order=self.order)
        out = targets[self.order]
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


def _broadcast_like(values: Array, reference: Array, /) -> Array:
    arr = jnp.asarray(values)
    ref = jnp.asarray(reference)
    if arr.ndim == ref.ndim:
        return arr
    if arr.ndim != 1:
        raise ValueError(
            f"Cannot broadcast shape {arr.shape} against reference shape {ref.shape}."
        )
    return arr.reshape((int(arr.shape[0]),) + (1,) * (ref.ndim - 1))


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
                return with_derivative_hook(base, _nearest_hook)
            raise ValueError(
                "TrajectorySignal with interpolation='nearest' is not differentiable; "
                "use interpolation='linear' for time derivatives."
            )

        return with_derivative_hook(base, _nearest_hook)

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
                return with_derivative_hook(base, _make_hook(0))
            out = DomainFunction(
                domain=domain,
                deps=domain.labels,
                func=_IrregularInterpolatedTrajectorySignal(table=table, order=n),
                metadata={},
            )
            return with_derivative_hook(out, _make_hook(n))

        return _hook

    return with_derivative_hook(base, _make_hook(0))


def TrajectorySignal(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
    *,
    interpolation: TrajectorySignalInterpolation = "linear",
    time_var: str | None = None,
    snap_tol: float = 1e-10,
) -> DomainFunction:
    """Expose fixed ragged trajectory data as a DomainFunction over `(data, t)`."""
    if not isinstance(domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)):
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
                return with_derivative_hook(base, _nearest_hook)
            raise ValueError(
                "TrajectorySignal with interpolation='nearest' is not differentiable; "
                "use interpolation='linear' or 'cubic_hermite' for time derivatives."
            )

        return with_derivative_hook(base, _nearest_hook)

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
                return with_derivative_hook(base, _make_hook(0))
            out = DomainFunction(
                domain=domain,
                deps=domain.labels,
                func=_InterpolatedTrajectorySignal(table=table, order=n),
                metadata={},
            )
            return with_derivative_hook(out, _make_hook(n))

        return _hook

    return with_derivative_hook(base, _make_hook(0))


__all__ = [
    "TrajectoryCaseDataBatch",
    "TrajectoryCaseDataConstraint",
    "TrajectoryCaseTime",
    "TrajectorySignal",
    "TrajectorySignalInterpolation",
]
