#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    DomainComponent,
    DomainFunction,
    PointBatch,
    PointSampling,
    TrajectoryDatasetDomain,
)

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..domain._trajectory_signal import TrajectorySignal
from ._data_metrics import (
    case_sample_count,
    normalize_case_sampling,
    reduce_supervised_loss,
    sample_case_indices as _sample_case_indices_uniform,
    supervised_data_metrics,
    supervised_per_sample_squared_error,
    validate_case_indices,
    validate_supervised_targets,
)


TrajectorySignalInterpolation = Literal["nearest", "linear", "cubic_hermite"]
TrajectoryCaseTime = Literal["start", "end"] | float


class TrajectoryCaseDataBatch(StrictModule):
    """A sampled mini-batch of case-level trajectory-domain data."""

    points: PointBatch
    target: Array
    case_indices: Array
    times: Array

    def __init__(
        self,
        *,
        points: PointBatch,
        target: ArrayLike,
        case_indices: ArrayLike,
        times: ArrayLike,
    ):
        self.points = points
        self.target = jnp.asarray(target)
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


class TrajectoryCaseDataTerm(AbstractSamplingTerm):
    """Supervise per-case targets on a TrajectoryDatasetDomain.

    The target has one row per trajectory case, for example scalar material
    parameters, class logits, or final scalar outputs attached to each dataset row.
    Case-only functions such as `theta(data)` ignore the representative time carried
    by the sampled trajectory batch; trajectory functions `theta(data, t)` are
    evaluated at the configured `case_time`.
    """

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
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
        field: str,
        component: DomainComponent,
        values: ArrayLike,
        /,
        *,
        sampling: PointSampling,
        case_time: TrajectoryCaseTime = "start",
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        case_indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        """Create a supervised case-level trajectory data constraint.

        Parameters:
            field: Name of the predicted function to supervise.
            component: Component from a `TrajectoryDatasetDomain`.
            values: Case targets with leading size equal to `domain.size`.
            sampling: Uniform empirical-case sampling plan.
            case_time: Representative time used if the predicted function depends
                on the trajectory time label.
            weight: Scalar or pointwise multiplier applied to this loss term.
            reduction: `"mean"` or `"sum"` over sampled cases.
            case_indices: Optional case subset for train/validation splits.
            label: Optional diagnostic label for this constraint.
            data_accuracy_eps: Stabilizer used in supervised data metrics.
        """
        if not isinstance(component.domain, TrajectoryDatasetDomain):
            raise TypeError(
                "TrajectoryCaseDataTerm requires a TrajectoryDatasetDomain component."
            )
        sampling_ = normalize_case_sampling(
            sampling,
            labels=component.domain.labels,
            owner="TrajectoryCaseDataTerm",
        )
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

        self.fields = (str(field),)
        self.component = component
        self.sampling = sampling_
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
            raise TypeError("TrajectoryCaseDataTerm domain is not a trajectory domain.")
        return domain

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        """Draw a case mini-batch and return aligned model inputs and targets."""
        domain = self.domain
        case_indices = _sample_case_indices(
            domain,
            case_sample_count(self.sampling),
            key,
            self.case_time,
            indices=self.case_indices,
        )
        times, time_indices = _time_selection(domain, case_indices, self.case_time)
        layout = self.sampling.layout
        assert layout is not None
        points = domain.points_from_case_time(
            case_indices,
            times,
            structure=layout,
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
        var = self.fields[0]
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
        """Return supervised diagnostics on a sampled or provided batch."""
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
        iter_: int | Array | None = None,
        batch: TrajectoryCaseDataBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return the weighted supervised squared-error loss."""
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


__all__ = [
    "TrajectoryCaseDataBatch",
    "TrajectoryCaseDataTerm",
    "TrajectoryCaseTime",
    "TrajectorySignal",
    "TrajectorySignalInterpolation",
]
