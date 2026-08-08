#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    DomainComponent,
    DomainFunction,
    PointBatch,
    PointSampling,
    RaggedSeriesDatasetDomain,
    RaggedSeriesSampling,
)

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ._data_metrics import (
    case_sample_count,
    normalize_case_sampling,
    reduce_supervised_loss,
    sample_case_indices,
    supervised_data_metrics,
    supervised_per_sample_squared_error,
    validate_case_indices,
    validate_supervised_targets,
)


class RaggedSeriesSupervisedBatch(StrictModule):
    """A sampled mini-batch of row-aligned ragged-series supervised data."""

    points: PointBatch
    target: Array
    indices: Array

    def __init__(
        self,
        *,
        points: PointBatch,
        target: ArrayLike,
        indices: ArrayLike,
    ):
        self.points = points
        self.target = jnp.asarray(target, dtype=float)
        self.indices = jnp.asarray(indices, dtype=jnp.int32)


def _validate_targets(domain: RaggedSeriesDatasetDomain, values: ArrayLike, /) -> Array:
    return validate_supervised_targets(
        values,
        leading_size=domain.size,
        name="ragged series supervised target",
    )


class RaggedSeriesSupervisedTerm(AbstractSamplingTerm):
    """Supervise row-aligned targets on a `RaggedSeriesDatasetDomain`.

    Each sampled case contributes one target row. The model receives the
    ragged-series payload for that case, either as the full padded row or as a
    fixed-width sampled view for efficient training on long records.
    """

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "sum"]
    series_sampling: RaggedSeriesSampling
    num_series_points: int | None
    values: Array
    weight: Array
    pointwise_weight: DomainFunction | None
    indices: Array | None
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
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        series_sampling: RaggedSeriesSampling = "full",
        num_series_points: int | None = None,
        indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        """Create a ragged-series supervised data constraint.

        Parameters:
            field: Name of the predicted function to supervise.
            component: Component from a `RaggedSeriesDatasetDomain`.
            values: Targets with leading size equal to `domain.size`.
            sampling: Uniform empirical-case sampling plan.
            weight: Scalar or pointwise multiplier applied to this loss term.
            reduction: `"mean"` for case-average loss or `"sum"` for a summed
                squared-error loss.
            series_sampling: `"full"` for full padded rows, or a sampled view mode
                such as `"points_uniform"`, `"window_uniform"`, `"prefix"`, or
                `"suffix"`.
            num_series_points: Width of sampled series views when
                `series_sampling` is not `"full"`.
            indices: Optional case subset for train/validation splits.
            label: Optional diagnostic label for this constraint.
            data_accuracy_eps: Stabilizer used in supervised data metrics.
        """
        if not isinstance(component.domain, RaggedSeriesDatasetDomain):
            raise TypeError(
                "RaggedSeriesSupervisedTerm requires a "
                "RaggedSeriesDatasetDomain component."
            )
        sampling_ = normalize_case_sampling(
            sampling,
            labels=component.domain.labels,
            owner="RaggedSeriesSupervisedTerm",
        )
        reduction_str = str(reduction)
        if reduction_str not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        reduction_value: Literal["mean", "sum"]
        if reduction_str == "mean":
            reduction_value = "mean"
        else:
            reduction_value = "sum"
        series_sampling_str = str(series_sampling)
        if series_sampling_str not in (
            "full",
            "points_uniform",
            "window_uniform",
            "prefix",
            "suffix",
        ):
            raise ValueError(
                "series_sampling must be 'full', 'points_uniform', "
                "'window_uniform', 'prefix', or 'suffix'."
            )
        series_sampling_value: RaggedSeriesSampling
        if series_sampling_str == "full":
            series_sampling_value = "full"
        elif series_sampling_str == "points_uniform":
            series_sampling_value = "points_uniform"
        elif series_sampling_str == "window_uniform":
            series_sampling_value = "window_uniform"
        elif series_sampling_str == "prefix":
            series_sampling_value = "prefix"
        else:
            series_sampling_value = "suffix"
        if series_sampling_value == "full":
            n_series_points = None
        else:
            if num_series_points is None:
                raise ValueError(
                    "num_series_points is required when series_sampling is not 'full'."
                )
            n_series_points = int(num_series_points)
            if n_series_points <= 0:
                raise ValueError("num_series_points must be positive.")

        domain = component.domain
        self.fields = (str(field),)
        self.component = component
        self.sampling = sampling_
        self.over = None
        self.reduction = reduction_value
        self.series_sampling = series_sampling_value
        self.num_series_points = n_series_points
        self.values = _validate_targets(domain, values)
        if isinstance(weight, DomainFunction):
            self.weight = jnp.asarray(1.0, dtype=float)
            self.pointwise_weight = weight
        else:
            self.weight = jnp.asarray(weight, dtype=float)
            self.pointwise_weight = None
        self.indices = validate_case_indices(
            indices,
            size=domain.size,
            name="indices",
        )
        self.label = None if label is None else str(label)
        self.data_accuracy_eps = jnp.asarray(float(data_accuracy_eps), dtype=float)

    @classmethod
    def bucketed(
        cls,
        field: str,
        component: DomainComponent,
        values: ArrayLike,
        /,
        *,
        sampling: PointSampling,
        num_buckets: int = 8,
        length_bucket_edges: ArrayLike | None = None,
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ) -> tuple["RaggedSeriesSupervisedTerm", ...]:
        """Build fixed-width full-series constraints grouped by valid length.

        Each returned constraint samples only from one length bucket and uses
        `series_sampling="prefix"` with `num_series_points` equal to that bucket's
        maximum valid length. This covers full sequences within the bucket while
        avoiding global max-length materialization during training. The plan's
        sampling count is split across buckets by bucket population, and each
        bucket loss is weighted by its population fraction to match the
        case-average objective of one full padded constraint.
        """
        if not isinstance(component.domain, RaggedSeriesDatasetDomain):
            raise TypeError(
                "RaggedSeriesSupervisedTerm.bucketed requires a "
                "RaggedSeriesDatasetDomain component."
            )
        sampling_ = normalize_case_sampling(
            sampling,
            labels=component.domain.labels,
            owner="RaggedSeriesSupervisedTerm.bucketed",
        )
        n = case_sample_count(sampling_)
        reduction_str = str(reduction)
        if reduction_str not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        domain = component.domain
        selected = validate_case_indices(
            indices,
            size=domain.size,
            name="indices",
        )
        if selected is None:
            case_indices = np.arange(domain.size, dtype=np.int32)
        else:
            case_indices = np.asarray(selected, dtype=np.int32)
        lengths = np.asarray(domain.lengths, dtype=np.int32)[case_indices]

        if length_bucket_edges is None:
            bucket_groups = _balanced_length_bucket_groups(
                case_indices,
                lengths,
                num_buckets=min(int(num_buckets), n),
            )
        else:
            bucket_groups = _edge_length_bucket_groups(
                case_indices,
                lengths,
                length_bucket_edges=length_bucket_edges,
            )
            if len(bucket_groups) > n:
                raise ValueError(
                    "num_cases must be at least the number of non-empty length "
                    "buckets."
                )

        total = float(case_indices.shape[0])
        bucket_sizes = np.asarray(
            [bucket_indices.shape[0] for bucket_indices, _width in bucket_groups],
            dtype=np.int32,
        )
        bucket_case_counts = _bucket_case_counts(
            num_cases=n,
            bucket_sizes=bucket_sizes,
        )
        constraints: list[RaggedSeriesSupervisedTerm] = []
        for bucket_id, ((bucket_indices, bucket_width), bucket_num_cases) in enumerate(
            zip(bucket_groups, bucket_case_counts, strict=True),
            start=1,
        ):
            if label is None:
                bucket_label = None
            else:
                bucket_label = f"{label}_bucket_{bucket_id}"
            bucket_fraction = bucket_indices.shape[0] / total
            if reduction_str == "sum":
                bucket_fraction *= n / int(bucket_num_cases)
            bucket_weight = _bucket_weight(weight, bucket_fraction)
            constraints.append(
                cls(
                    field,
                    component,
                    values,
                    sampling=PointSampling(
                        int(bucket_num_cases),
                        layout=sampling_.layout,
                        design=sampling_.design,
                    ),
                    weight=bucket_weight,
                    reduction=reduction,
                    series_sampling="prefix",
                    num_series_points=bucket_width,
                    indices=jnp.asarray(bucket_indices, dtype=jnp.int32),
                    label=bucket_label,
                    data_accuracy_eps=data_accuracy_eps,
                )
            )
        return tuple(constraints)

    @property
    def domain(self) -> RaggedSeriesDatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, RaggedSeriesDatasetDomain):
            raise TypeError(
                "RaggedSeriesSupervisedTerm domain is not a "
                "RaggedSeriesDatasetDomain."
            )
        return domain

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        """Draw a case mini-batch and return aligned model inputs and targets."""
        domain = self.domain
        key_cases = key
        key_series = key
        if self.series_sampling != "full":
            key_cases, key_series = jr.split(key, 2)
        indices = sample_case_indices(
            size=domain.size,
            num_samples=case_sample_count(self.sampling),
            key=key_cases,
            indices=self.indices,
        )
        layout = self.sampling.layout
        assert layout is not None
        if self.series_sampling == "full":
            points = domain.points_from_indices(indices, structure=layout)
        else:
            if self.num_series_points is None:
                raise ValueError("num_series_points is required for sampled series.")
            points = domain.sampled_points_from_indices(
                indices,
                num_series_points=self.num_series_points,
                sampling=self.series_sampling,
                structure=layout,
                key=key_series,
            )
        return RaggedSeriesSupervisedBatch(
            points=points,
            target=self.values[indices],
            indices=indices,
        )

    def _prediction(
        self,
        functions: Mapping[str, DomainFunction],
        batch: RaggedSeriesSupervisedBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        var = self.fields[0]
        prediction = functions[var](batch.points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError("Expected ragged series prediction to return a coordax.Field.")
        return prediction

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: RaggedSeriesSupervisedBatch | None = None,
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
        batch: RaggedSeriesSupervisedBatch | None = None,
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


def _balanced_length_bucket_groups(
    case_indices: np.ndarray,
    lengths: np.ndarray,
    /,
    *,
    num_buckets: int,
) -> tuple[tuple[np.ndarray, int], ...]:
    n_buckets = int(num_buckets)
    if n_buckets <= 0:
        raise ValueError("num_buckets must be positive.")
    n_buckets = min(n_buckets, int(case_indices.shape[0]))
    order = np.argsort(lengths, kind="stable")
    groups: list[tuple[np.ndarray, int]] = []
    for order_group in np.array_split(order, n_buckets):
        if order_group.size == 0:
            continue
        bucket_indices = np.asarray(case_indices[order_group], dtype=np.int32)
        bucket_width = int(np.max(lengths[order_group]))
        groups.append((bucket_indices, bucket_width))
    return tuple(groups)


def _bucket_case_counts(
    *,
    num_cases: int,
    bucket_sizes: np.ndarray,
) -> np.ndarray:
    n = int(num_cases)
    sizes = np.asarray(bucket_sizes, dtype=np.int32)
    if sizes.ndim != 1:
        raise ValueError("bucket_sizes must have shape (B,).")
    if int(sizes.shape[0]) <= 0:
        raise ValueError("bucket_sizes must be non-empty.")
    if np.any(sizes <= 0):
        raise ValueError("bucket_sizes must be positive.")
    if n < int(sizes.shape[0]):
        raise ValueError(
            "num_cases must be at least the number of non-empty length buckets."
        )

    probabilities = sizes.astype(float) / float(np.sum(sizes))
    raw = probabilities * float(n)
    counts = np.floor(raw).astype(np.int32)
    counts = np.maximum(counts, np.ones_like(counts))

    while int(np.sum(counts)) > n:
        candidates = np.flatnonzero(counts > 1)
        candidate = int(candidates[np.argmin(raw[candidates])])
        counts[candidate] -= 1

    remaining = n - int(np.sum(counts))
    if remaining > 0:
        fractional = raw - np.floor(raw)
        order = np.argsort(-fractional, kind="stable")
        for i in range(remaining):
            counts[int(order[i % int(order.shape[0])])] += 1

    return counts


def _edge_length_bucket_groups(
    case_indices: np.ndarray,
    lengths: np.ndarray,
    /,
    *,
    length_bucket_edges: ArrayLike,
) -> tuple[tuple[np.ndarray, int], ...]:
    edge_arr = np.asarray(length_bucket_edges)
    if edge_arr.ndim != 1:
        raise ValueError("length_bucket_edges must have shape (B,).")
    if int(edge_arr.shape[0]) <= 0:
        raise ValueError("length_bucket_edges must be non-empty.")
    edge_float = edge_arr.astype(float)
    edges = np.ceil(edge_float).astype(np.int32)
    if np.any(edge_float <= 0.0):
        raise ValueError("length_bucket_edges must be positive.")
    if np.any(edges.astype(float) != edge_float):
        raise ValueError("length_bucket_edges must contain integer lengths.")
    if np.any(edges[1:] <= edges[:-1]):
        raise ValueError("length_bucket_edges must be strictly increasing.")
    max_length = int(np.max(lengths))
    if int(edges[-1]) < max_length:
        raise ValueError(
            "length_bucket_edges must include the maximum selected series length."
        )

    groups: list[tuple[np.ndarray, int]] = []
    previous_edge = 0
    for edge in edges:
        in_bucket = (lengths > previous_edge) & (lengths <= int(edge))
        if np.any(in_bucket):
            bucket_indices = np.asarray(case_indices[in_bucket], dtype=np.int32)
            groups.append((bucket_indices, int(edge)))
        previous_edge = int(edge)
    return tuple(groups)


def _bucket_weight(
    weight: DomainFunction | ArrayLike,
    fraction: float,
    /,
) -> DomainFunction | Array:
    if isinstance(weight, DomainFunction):
        return weight * float(fraction)
    return jnp.asarray(weight, dtype=float) * float(fraction)


__all__ = [
    "RaggedSeriesSupervisedBatch",
    "RaggedSeriesSupervisedTerm",
]
