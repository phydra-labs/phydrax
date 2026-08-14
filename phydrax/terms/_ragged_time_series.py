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

from phydrax.domain import (
    DomainComponent,
    DomainFunction,
    IrregularTrajectoryDatasetDomain,
    PointBatch,
    PointSampling,
    SampleLayout,
    TRAJECTORY_CASE_INDEX_KEY,
    TRAJECTORY_TIME_INDEX_KEY,
    TrajectoryDatasetDomain,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._interpolation import (
    apply_gather_stencil,
    linear_stencil_from_indices,
    nearest_stencil_from_indices,
)
from .._sampling import design_name
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
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

    points: PointBatch
    target: Array
    case_indices: Array
    time_indices: Array
    times: Array

    def __init__(
        self,
        *,
        points: PointBatch,
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
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
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


def _validate_values(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
) -> Array:
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
    time_count = int(values.shape[1])
    source = values.reshape((-1,) + values.shape[2:])
    indices = case_indices * time_count + time_indices
    stencil = nearest_stencil_from_indices(
        indices,
        source_size=int(source.shape[0]),
    )
    return apply_gather_stencil(source, stencil).values


def _gather_linear(
    values: Array,
    case_indices: Array,
    tau: Array,
    lengths: Array,
    /,
) -> tuple[Array, Array]:
    lower = jnp.floor(tau).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, lengths - 1)
    fraction = tau - lower.astype(float)
    time_count = int(values.shape[1])
    source = values.reshape((-1,) + values.shape[2:])
    stencil = linear_stencil_from_indices(
        case_indices * time_count + lower,
        case_indices * time_count + upper,
        fraction,
        source_size=int(source.shape[0]),
    )
    return apply_gather_stencil(source, stencil).values, lower


def _gather_linear_irregular(
    domain: IrregularTrajectoryDatasetDomain,
    values: Array,
    case_indices: Array,
    times: Array,
    /,
) -> tuple[Array, Array]:
    lower, upper, fraction = domain.bracketing_time_indices(case_indices, times)
    time_count = int(values.shape[1])
    source = values.reshape((-1,) + values.shape[2:])
    stencil = linear_stencil_from_indices(
        case_indices * time_count + lower,
        case_indices * time_count + upper,
        fraction,
        source_size=int(source.shape[0]),
    )
    return apply_gather_stencil(source, stencil).values, lower


def _normalize_sampling(
    sampling: PointSampling,
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    /,
) -> PointSampling:
    if not isinstance(sampling, PointSampling):
        raise TypeError("RaggedTimeSeriesDataTerm requires a PointSampling plan.")
    count = sampling.count
    if isinstance(count, int):
        if count <= 0:
            raise ValueError("PointSampling.count must be positive.")
        default_layout = SampleLayout((domain.labels,))
    else:
        if len(count) != 2:
            raise ValueError(
                "RaggedTimeSeriesDataTerm requires count=(num_cases, num_times)."
            )
        if count[0] <= 0 or count[1] <= 0:
            raise ValueError("PointSampling count entries must be positive.")
        default_layout = SampleLayout(((domain.data_label,), (domain.time_label,)))
    if design_name(sampling.design) != "uniform":
        raise ValueError("RaggedTimeSeriesDataTerm supports only uniform sampling.")
    return PointSampling(
        count,
        layout=sampling.layout or default_layout,
        design=sampling.design,
    )


def _case_time_grid_structure(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    structure: SampleLayout,
    /,
) -> tuple[SampleLayout, str, str]:
    structure_ = structure.canonicalize(domain.labels)
    data_axis = structure_.axis_for(domain.data_label)
    time_axis = structure_.axis_for(domain.time_label)
    if data_axis is None or time_axis is None or data_axis == time_axis:
        raise ValueError(
            "case-major ragged trajectory batches require SampleLayout "
            "with separate singleton data and time blocks."
        )
    for block in structure_.blocks:
        if domain.data_label in block and len(block) != 1:
            raise ValueError("data_label must be sampled in a singleton block.")
        if domain.time_label in block and len(block) != 1:
            raise ValueError("time_label must be sampled in a singleton block.")
    return structure_, data_axis, time_axis


def _grid_points_from_case_time(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    case_indices: Array,
    times: Array,
    time_indices: Array,
    /,
    *,
    structure: SampleLayout,
) -> PointBatch:
    structure_, data_axis, time_axis = _case_time_grid_structure(domain, structure)
    case_idx = jnp.asarray(case_indices, dtype=jnp.int32).reshape((-1,))
    time_arr = jnp.asarray(times, dtype=float)
    time_idx = jnp.asarray(time_indices, dtype=jnp.int32)
    if time_arr.ndim != 2:
        raise ValueError("case-major times must have shape (num_cases, num_times).")
    if time_idx.shape != time_arr.shape:
        raise ValueError("time_indices must have the same shape as times.")
    if int(time_arr.shape[0]) != int(case_idx.shape[0]):
        raise ValueError("times leading axis must match case_indices length.")

    data_samples = domain.input_rows(case_idx)

    def _to_data_field(v: ArrayLike):
        arr = jnp.asarray(v)
        if arr.ndim == 0:
            raise ValueError("Trajectory input rows must retain a case axis.")
        return cx.Field(arr, dims=(data_axis,) + (None,) * (arr.ndim - 1))

    points: dict[str, Any] = {}
    points[domain.data_label] = jax.tree_util.tree_map(_to_data_field, data_samples)
    points[domain.time_label] = cx.Field(time_arr, dims=(data_axis, time_axis))
    points[TRAJECTORY_CASE_INDEX_KEY] = cx.Field(case_idx, dims=(data_axis,))
    points[TRAJECTORY_TIME_INDEX_KEY] = cx.Field(
        time_idx,
        dims=(data_axis, time_axis),
    )
    metadata = {
        "trajectory_data_axis": data_axis,
        "trajectory_time_axis": time_axis,
    }
    return PointBatch(
        points=frozendict(points),
        structure=structure_,
        metadata=metadata,
    )


def _gather_nearest_grid(
    values: Array,
    case_indices: Array,
    time_indices: Array,
    /,
) -> Array:
    return values[case_indices[:, None], time_indices]


def _gather_linear_grid(
    values: Array,
    case_indices: Array,
    tau: Array,
    lengths: Array,
    /,
) -> tuple[Array, Array]:
    flat_cases = jnp.broadcast_to(case_indices[:, None], tau.shape).reshape((-1,))
    flat_tau = tau.reshape((-1,))
    flat_lengths = jnp.broadcast_to(lengths[:, None], tau.shape).reshape((-1,))
    target, lower = _gather_linear(values, flat_cases, flat_tau, flat_lengths)
    target_shape = tau.shape + target.shape[1:]
    return target.reshape(target_shape), lower.reshape(tau.shape)


def _gather_linear_irregular_grid(
    domain: IrregularTrajectoryDatasetDomain,
    values: Array,
    case_indices: Array,
    times: Array,
    /,
) -> tuple[Array, Array]:
    flat_cases = jnp.broadcast_to(case_indices[:, None], times.shape).reshape((-1,))
    flat_times = times.reshape((-1,))
    target, lower = _gather_linear_irregular(domain, values, flat_cases, flat_times)
    target_shape = times.shape + target.shape[1:]
    return target.reshape(target_shape), lower.reshape(times.shape)


def _flatten_grid_prediction_target(
    prediction: Array,
    target: Array,
    batch: "RaggedTimeSeriesBatch",
    /,
) -> tuple[Array, Array]:
    pred_arr = jnp.asarray(prediction)
    target_arr = jnp.asarray(target)
    if batch.times.ndim != 2:
        return pred_arr, target_arr
    grid_shape = tuple(int(n) for n in batch.times.shape)
    n = grid_shape[0] * grid_shape[1]
    if pred_arr.shape[:2] == grid_shape:
        pred_arr = pred_arr.reshape((n,) + pred_arr.shape[2:])
    if target_arr.shape[:2] == grid_shape:
        target_arr = target_arr.reshape((n,) + target_arr.shape[2:])
    return pred_arr, target_arr


def _flatten_grid_weight(weight: Array, batch: "RaggedTimeSeriesBatch", /) -> Array:
    weight_arr = jnp.asarray(weight, dtype=float)
    if batch.times.ndim != 2 or weight_arr.ndim == 0:
        return weight_arr
    n_cases = int(batch.times.shape[0])
    n_times = int(batch.times.shape[1])
    n = n_cases * n_times
    grid_shape = (n_cases, n_times)
    if weight_arr.shape[:2] == grid_shape:
        return weight_arr.reshape((n,) + weight_arr.shape[2:])
    if int(weight_arr.shape[0]) == n_cases:
        expanded = jnp.broadcast_to(
            weight_arr[:, None],
            grid_shape + weight_arr.shape[1:],
        )
        return expanded.reshape((n,) + weight_arr.shape[1:])
    return weight_arr


class RaggedTimeSeriesDataTerm(AbstractSamplingTerm):
    """Supervised data constraint for ragged trajectories on a TrajectoryDatasetDomain.

    This constraint samples observed or interpolated `(data, t)` pairs, evaluates a
    model function at those paired points, and penalizes disagreement with the stored
    ragged time-series targets.
    """

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
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
    selection: RaggedTimeSeriesSampling
    interpolation: RaggedTimeSeriesInterpolation
    data_accuracy_eps: Array

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        values: ArrayLike,
        /,
        *,
        sampling: PointSampling,
        selection: RaggedTimeSeriesSampling = "observation_uniform",
        interpolation: RaggedTimeSeriesInterpolation = "nearest",
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        case_indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        """Create a supervised ragged time-series data constraint.

        Parameters:
            field: Name of the predicted function to supervise.
            component: Component from a trajectory dataset domain.
            values: Padded target array with one leading row per case and a time
                axis at position 1.
            sampling: Uniform point plan. A scalar count requests paired
                observations; `(num_cases, num_times)` requests a case-time grid.
            selection: Observation-level, case-level, or continuous case-time mode.
            interpolation: Target lookup mode for continuous time sampling.
            weight: Scalar or pointwise multiplier applied to this loss term.
            reduction: `"mean"` or `"sum"` over sampled observations.
            case_indices: Optional case subset for train/validation splits.
            label: Optional diagnostic label for this constraint.
            data_accuracy_eps: Stabilizer used in supervised data metrics.
        """
        if not isinstance(
            component.domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
        ):
            raise TypeError(
                "RaggedTimeSeriesDataTerm requires a trajectory dataset domain component."
            )
        sampling_ = _normalize_sampling(sampling, component.domain)

        sampling_str = str(selection)
        if sampling_str not in (
            "observation_uniform",
            "case_uniform",
            "case_time_uniform",
        ):
            raise ValueError(
                "selection must be 'observation_uniform', 'case_uniform', "
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
        self.fields = (str(field),)
        self.component = component
        self.sampling = sampling_
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
        self.selection = sampling_value
        self.interpolation = interpolation_value
        self.data_accuracy_eps = jnp.asarray(float(data_accuracy_eps), dtype=float)

    @property
    def domain(self) -> TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain:
        domain = self.component.domain
        if not isinstance(
            domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
        ):
            raise TypeError("RaggedTimeSeriesDataTerm domain is not a trajectory domain.")
        return domain

    @property
    def _layout(self) -> SampleLayout:
        layout = self.sampling.layout
        assert layout is not None
        return layout

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        """Draw a ragged trajectory mini-batch and aligned target values."""
        domain = self.domain
        count = self.sampling.count
        if isinstance(count, tuple):
            return self._sample_case_time_grid(domain, key=key)

        n = count
        key_case, key_time = jr.split(key)

        if self.selection == "observation_uniform":
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
            if self.selection == "case_time_uniform":
                if isinstance(domain, IrregularTrajectoryDatasetDomain):
                    start_times = domain.start_times[case_indices]
                    end_times = domain.end_times[case_indices]
                    tau = start_times + jr.uniform(key_time, shape=(n,)) * (
                        end_times - start_times
                    )
                    if self.interpolation == "linear":
                        target, time_indices = _gather_linear_irregular(
                            domain, self.values, case_indices, tau
                        )
                    else:
                        time_indices = domain.nearest_time_indices(case_indices, tau)
                        target = _gather_nearest(self.values, case_indices, time_indices)
                    times = tau
                else:
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
            structure=self._layout,
            time_indices=time_indices,
        )
        return RaggedTimeSeriesBatch(
            points=points,
            target=target,
            case_indices=case_indices,
            time_indices=time_indices,
            times=times,
        )

    def _sample_case_time_grid(
        self,
        domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> RaggedTimeSeriesBatch:
        count = self.sampling.count
        if not isinstance(count, tuple):
            raise TypeError("case-time grid sampling requires a tuple point count.")
        n_cases, n_times = count
        key_case, key_time = jr.split(key)
        case_indices = _sample_case_indices(
            size=domain.size,
            num_samples=n_cases,
            key=key_case,
            indices=self.case_indices,
        )
        lengths = domain.lengths[case_indices]
        case_grid = jnp.broadcast_to(case_indices[:, None], (n_cases, n_times))

        if self.selection == "case_time_uniform":
            if isinstance(domain, IrregularTrajectoryDatasetDomain):
                start_times = domain.start_times[case_indices]
                end_times = domain.end_times[case_indices]
                u = jr.uniform(key_time, shape=(n_cases, n_times))
                times = start_times[:, None] + u * (
                    end_times[:, None] - start_times[:, None]
                )
                if self.interpolation == "linear":
                    target, time_indices = _gather_linear_irregular_grid(
                        domain,
                        self.values,
                        case_indices,
                        times,
                    )
                else:
                    time_indices = domain.nearest_time_indices(
                        case_grid.reshape((-1,)),
                        times.reshape((-1,)),
                    ).reshape((n_cases, n_times))
                    target = _gather_nearest_grid(
                        self.values,
                        case_indices,
                        time_indices,
                    )
            else:
                u = jr.uniform(key_time, shape=(n_cases, n_times))
                tau = u * (lengths[:, None].astype(float) - 1.0)
                if self.interpolation == "linear":
                    target, time_indices = _gather_linear_grid(
                        self.values,
                        case_indices,
                        tau,
                        lengths,
                    )
                else:
                    time_indices = jnp.rint(tau).astype(jnp.int32)
                    time_indices = jnp.clip(time_indices, 0, lengths[:, None] - 1)
                    target = _gather_nearest_grid(
                        self.values,
                        case_indices,
                        time_indices,
                    )
                times = domain.start + domain.dt * tau
        else:
            u = jr.uniform(key_time, shape=(n_cases, n_times))
            time_indices = jnp.floor(u * lengths[:, None].astype(float)).astype(jnp.int32)
            time_indices = jnp.clip(time_indices, 0, lengths[:, None] - 1)
            times = domain.observation_times(case_grid, time_indices)
            target = _gather_nearest_grid(self.values, case_indices, time_indices)

        points = _grid_points_from_case_time(
            domain,
            case_indices,
            times,
            time_indices,
            structure=self._layout,
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
        var = self.fields[0]
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
        """Return supervised diagnostics on a sampled or provided batch."""
        batch_ = self.sample(key=key) if batch is None else batch
        prediction = self._prediction(functions, batch_, key=key, **kwargs)
        pred_arr, target_arr = _flatten_grid_prediction_target(
            jnp.asarray(prediction.data, dtype=float),
            batch_.target,
            batch_,
        )
        return supervised_data_metrics(
            pred_arr,
            target_arr,
            eps=self.data_accuracy_eps,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: RaggedTimeSeriesBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return the weighted supervised squared-error loss."""
        del iter_
        batch_ = self.sample(key=key) if batch is None else batch
        prediction = self._prediction(functions, batch_, key=key, **kwargs)
        pred_arr, target_arr = _flatten_grid_prediction_target(
            jnp.asarray(prediction.data, dtype=float),
            batch_.target,
            batch_,
        )
        per_sample = supervised_per_sample_squared_error(
            pred_arr,
            target_arr,
        )

        if self.pointwise_weight is not None:
            w = self.pointwise_weight(batch_.points, key=key, **kwargs)
            if not isinstance(w, cx.Field):
                raise TypeError("pointwise weight must return a coordax.Field.")
            w_arr = _flatten_grid_weight(jnp.asarray(w.data, dtype=float), batch_)
            if w_arr.ndim == 0:
                per_sample = per_sample * w_arr
            else:
                per_sample = per_sample * jnp.squeeze(w_arr).reshape((-1,))

        reduced = reduce_supervised_loss(per_sample, reduction=self.reduction)
        return self.weight * jnp.asarray(reduced, dtype=float).reshape(())


__all__ = [
    "RaggedTimeSeriesBatch",
    "RaggedTimeSeriesDataTerm",
    "RaggedTimeSeriesInterpolation",
    "RaggedTimeSeriesSampling",
]
