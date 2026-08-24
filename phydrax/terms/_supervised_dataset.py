#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    DatasetDomain,
    DomainComponent,
    DomainFunction,
    PointBatch,
    PointSampling,
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


class SupervisedDatasetBatch(StrictModule):
    """A sampled mini-batch of row-aligned supervised dataset data."""

    points: PointBatch
    target: Array
    indices: Array
    sample_weight: Array | None

    def __init__(
        self,
        *,
        points: PointBatch,
        target: ArrayLike,
        indices: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ):
        target_array = jnp.asarray(target)
        indices_array = jnp.asarray(indices, dtype=jnp.int32)
        if indices_array.ndim != 1:
            raise ValueError("Supervised dataset batch indices must be one-dimensional.")
        if target_array.ndim == 0 or int(target_array.shape[0]) != int(
            indices_array.shape[0]
        ):
            raise ValueError(
                "Supervised dataset batch targets must retain the sampled case axis."
            )
        if sample_weight is None:
            weight_array = None
        else:
            weight_array = jnp.asarray(sample_weight, dtype=float)
            if weight_array.shape != indices_array.shape:
                raise ValueError(
                    "Supervised dataset batch sample weights must match its indices."
                )
            if bool(jnp.any(~jnp.isfinite(weight_array))) or bool(
                jnp.any(weight_array <= 0.0)
            ):
                raise ValueError(
                    "Supervised dataset batch sample weights must be finite and "
                    "strictly positive."
                )
        self.points = points
        self.target = target_array
        self.indices = indices_array
        self.sample_weight = weight_array


def _validate_targets(domain: DatasetDomain, values: ArrayLike, /) -> Array:
    return validate_supervised_targets(
        values,
        leading_size=domain.size,
        name="supervised dataset target",
    )


class SupervisedDatasetTerm(AbstractSamplingTerm):
    """Supervise row-aligned targets on a DatasetDomain.

    `DatasetDomain` owns empirical row payloads. This term owns aligned targets
    and samples row indices so predictions and targets stay synchronized.
    """

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "sum"]
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
        indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        if not isinstance(component.domain, DatasetDomain):
            raise TypeError("SupervisedDatasetTerm requires a DatasetDomain component.")
        sampling_ = normalize_case_sampling(
            sampling,
            labels=component.domain.labels,
            owner="SupervisedDatasetTerm",
        )
        reduction_value = str(reduction)
        if reduction_value not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        reduction_: Literal["mean", "sum"] = (
            "mean" if reduction_value == "mean" else "sum"
        )

        domain = component.domain
        self.fields = (str(field),)
        self.component = component
        self.sampling = sampling_
        self.over = None
        self.reduction = reduction_
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

    @property
    def domain(self) -> DatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, DatasetDomain):
            raise TypeError("SupervisedDatasetTerm domain is not a DatasetDomain.")
        return domain

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        domain = self.domain
        indices = sample_case_indices(
            size=domain.size,
            num_samples=case_sample_count(self.sampling),
            key=key,
            indices=self.indices,
        )
        layout = self.sampling.layout
        assert layout is not None
        points = domain.points_from_indices(indices, structure=layout)
        return SupervisedDatasetBatch(
            points=points,
            target=self.values[indices],
            indices=indices,
        )

    def _prediction(
        self,
        functions: Mapping[str, DomainFunction],
        batch: SupervisedDatasetBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        var = self.fields[0]
        prediction = functions[var](batch.points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError("Expected dataset prediction to return a coordax.Field.")
        return prediction

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch | None = None,
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
        iter_: int | Array | None = None,
        batch: SupervisedDatasetBatch | None = None,
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


__all__ = ["SupervisedDatasetBatch", "SupervisedDatasetTerm"]
