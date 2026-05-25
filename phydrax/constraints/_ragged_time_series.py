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

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ..domain._components import DomainComponent
from ..domain._function import DomainFunction
from ..domain._structure import PointsBatch, ProductStructure
from ..domain._trajectory_dataset import TrajectoryDatasetDomain
from ._base import AbstractSamplingConstraint
from ._data_metrics import (
    reduce_supervised_loss,
    sample_case_indices as _sample_case_indices,
    supervised_data_metrics,
    supervised_per_sample_squared_error,
    validate_case_indices,
)


RaggedTimeSeriesSampling = Literal[
    "observation_uniform",
    "case_uniform",
    "case_time_uniform",
]
RaggedTimeSeriesInterpolation = Literal["nearest", "linear"]


class RaggedTimeSeriesBatch(StrictModule):
    """A sampled mini-batch from a ragged trajectory dataset."""

    points: PointsBatch
    target: Array
    case_indices: Array
    time_indices: Array
    times: Array

    def __init__(
        self,
        *,
        points: PointsBatch,
        target: ArrayLike,
        case_indices: ArrayLike,
        time_indices: ArrayLike,
        times: ArrayLike,
    ):
        self.points = points
        self.target = jnp.asarray(target, dtype=float)
        self.case_indices = jnp.asarray(case_indices, dtype=jnp.int32)
        self.time_indices = jnp.asarray(time_indices, dtype=jnp.int32)
        self.times = jnp.asarray(times, dtype=float)


def _flat_observation_indices(
    domain: TrajectoryDatasetDomain,
    case_indices: Array | None = None,
    /,
) -> tuple[Array, Array]:
    flat_cases = domain.flat_case_indices
    flat_times = domain.flat_time_indices
    if case_indices is None:
        return flat_cases, flat_times
    allowed = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
    mask = jnp.any(flat_cases[:, None] == allowed[None, :], axis=1)
    cases = flat_cases[mask]
    times = flat_times[mask]
    if int(cases.shape[0]) <= 0:
        raise ValueError("case_indices contain no valid trajectory observations.")
    return cases, times


def _validate_values(domain: TrajectoryDatasetDomain, values: ArrayLike, /) -> Array:
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim < 2:
        raise ValueError(
            "Ragged time-series values must have shape (N, T, ...) with a time axis."
        )
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


def _gather_nearest(values: Array, case_indices: Array, time_indices: Array, /) -> Array:
    return values[case_indices, time_indices]


def _gather_linear(
    values: Array,
    case_indices: Array,
    tau: Array,
    lengths: Array,
    /,
) -> tuple[Array, Array]:
    lower = jnp.floor(tau).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, lengths - 1)
    frac = tau - lower.astype(float)
    lower_values = values[case_indices, lower]
    upper_values = values[case_indices, upper]
    frac_shape = (int(frac.shape[0]),) + (1,) * max(values.ndim - 2, 0)
    frac_w = frac.reshape(frac_shape)
    target = (1.0 - frac_w) * lower_values + frac_w * upper_values
    return target, lower


class RaggedTimeSeriesDataConstraint(AbstractSamplingConstraint):
    """Supervised data constraint for ragged trajectories on a TrajectoryDatasetDomain.

    This constraint samples observed or interpolated `(data, t)` pairs, evaluates a
    model function at those paired points, and penalizes disagreement with the stored
    ragged time-series targets.
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
    weight: Array
    pointwise_weight: DomainFunction | None
    case_indices: Array | None
    observation_case_indices: Array
    observation_time_indices: Array
    observation_count: int
    label: str | None
    sampling: RaggedTimeSeriesSampling
    interpolation: RaggedTimeSeriesInterpolation
    data_accuracy_eps: Array

    def __init__(
        self,
        constraint_var: str,
        component: DomainComponent,
        values: ArrayLike,
        /,
        *,
        num_points: int,
        structure: ProductStructure | None = None,
        sampling: RaggedTimeSeriesSampling = "observation_uniform",
        interpolation: RaggedTimeSeriesInterpolation = "nearest",
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        case_indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        if not isinstance(component.domain, TrajectoryDatasetDomain):
            raise TypeError(
                "RaggedTimeSeriesDataConstraint requires a TrajectoryDatasetDomain component."
            )
        n = int(num_points)
        if n <= 0:
            raise ValueError("num_points must be positive.")

        sampling_str = str(sampling)
        if sampling_str not in (
            "observation_uniform",
            "case_uniform",
            "case_time_uniform",
        ):
            raise ValueError(
                "sampling must be 'observation_uniform', 'case_uniform', "
                "or 'case_time_uniform'."
            )
        sampling_value: RaggedTimeSeriesSampling
        if sampling_str == "observation_uniform":
            sampling_value = "observation_uniform"
        elif sampling_str == "case_uniform":
            sampling_value = "case_uniform"
        else:
            sampling_value = "case_time_uniform"

        interpolation_str = str(interpolation)
        if interpolation_str not in ("nearest", "linear"):
            raise ValueError("interpolation must be either 'nearest' or 'linear'.")
        interpolation_value: RaggedTimeSeriesInterpolation
        if interpolation_str == "nearest":
            interpolation_value = "nearest"
        else:
            interpolation_value = "linear"

        reduction_str = str(reduction)
        if reduction_str not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        reduction_value: Literal["mean", "sum"]
        if reduction_str == "mean":
            reduction_value = "mean"
        else:
            reduction_value = "sum"

        domain = component.domain
        self.constraint_vars = (str(constraint_var),)
        self.component = component
        self.structure = structure or ProductStructure((domain.labels,))
        self.dense_structure = None
        self.num_points = n
        self.sampler = sampling_value
        self.over = None
        self.reduction = reduction_value
        self.values = _validate_values(domain, values)
        if isinstance(weight, DomainFunction):
            self.weight = jnp.asarray(1.0, dtype=float)
            self.pointwise_weight = weight
        else:
            self.weight = jnp.asarray(weight, dtype=float)
            self.pointwise_weight = None
        self.case_indices = validate_case_indices(
            case_indices,
            size=domain.size,
            name="case_indices",
        )
        obs_cases, obs_times = _flat_observation_indices(domain, self.case_indices)
        self.observation_case_indices = obs_cases
        self.observation_time_indices = obs_times
        self.observation_count = int(obs_cases.shape[0])
        self.label = None if label is None else str(label)
        self.sampling = sampling_value
        self.interpolation = interpolation_value
        self.data_accuracy_eps = jnp.asarray(float(data_accuracy_eps), dtype=float)

    @property
    def domain(self) -> TrajectoryDatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, TrajectoryDatasetDomain):
            raise TypeError(
                "RaggedTimeSeriesDataConstraint domain is not a trajectory domain."
            )
        return domain

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        domain = self.domain
        n = int(self.num_points)
        key_case, key_time = jr.split(key)

        if self.sampling == "observation_uniform":
            flat_cases = self.observation_case_indices
            flat_times = self.observation_time_indices
            obs_idx = jr.randint(
                key,
                shape=(n,),
                minval=0,
                maxval=self.observation_count,
                dtype=jnp.int32,
            )
            case_indices = flat_cases[obs_idx]
            time_indices = flat_times[obs_idx]
            times = domain.observation_times(case_indices, time_indices)
            target = _gather_nearest(self.values, case_indices, time_indices)
        else:
            case_indices = _sample_case_indices(
                size=domain.size,
                num_samples=n,
                key=key_case,
                indices=self.case_indices,
            )
            lengths = domain.lengths[case_indices]
            if self.sampling == "case_time_uniform":
                tau = jr.uniform(key_time, shape=(n,)) * (lengths.astype(float) - 1.0)
                if self.interpolation == "linear":
                    target, time_indices = _gather_linear(
                        self.values, case_indices, tau, lengths
                    )
                else:
                    time_indices = jnp.rint(tau).astype(jnp.int32)
                    time_indices = jnp.clip(time_indices, 0, lengths - 1)
                    target = _gather_nearest(self.values, case_indices, time_indices)
                times = domain.start + domain.dt * tau
            else:
                u = jr.uniform(key_time, shape=(n,))
                time_indices = jnp.floor(u * lengths.astype(float)).astype(jnp.int32)
                time_indices = jnp.clip(time_indices, 0, lengths - 1)
                times = domain.observation_times(case_indices, time_indices)
                target = _gather_nearest(self.values, case_indices, time_indices)

        points = domain.points_from_case_time(
            case_indices,
            times,
            structure=self.structure,
            time_indices=time_indices,
        )
        return RaggedTimeSeriesBatch(
            points=points,
            target=target,
            case_indices=case_indices,
            time_indices=time_indices,
            times=times,
        )

    def _prediction(
        self,
        functions: Mapping[str, DomainFunction],
        batch: RaggedTimeSeriesBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        var = self.constraint_vars[0]
        prediction = functions[var](batch.points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError("Expected data prediction to return a coordax.Field.")
        return prediction

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: RaggedTimeSeriesBatch | None = None,
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
        batch: RaggedTimeSeriesBatch | None = None,
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


__all__ = [
    "RaggedTimeSeriesBatch",
    "RaggedTimeSeriesDataConstraint",
    "RaggedTimeSeriesInterpolation",
    "RaggedTimeSeriesSampling",
]
