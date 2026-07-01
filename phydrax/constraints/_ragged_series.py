#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ..domain._components import DomainComponent
from ..domain._function import DomainFunction
from ..domain._ragged_series_dataset import RaggedSeriesDatasetDomain
from ..domain._structure import PointsBatch, ProductStructure
from ._base import AbstractSamplingConstraint
from ._data_metrics import (
    reduce_supervised_loss,
    sample_case_indices,
    supervised_data_metrics,
    supervised_per_sample_squared_error,
    validate_case_indices,
    validate_supervised_targets,
)


class RaggedSeriesSupervisedBatch(StrictModule):
    """A sampled mini-batch of row-aligned ragged-series supervised data."""

    points: PointsBatch
    target: Array
    indices: Array

    def __init__(
        self,
        *,
        points: PointsBatch,
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


class RaggedSeriesSupervisedConstraint(AbstractSamplingConstraint):
    """Supervise row-aligned targets on a `RaggedSeriesDatasetDomain`."""

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
    indices: Array | None
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
        sampler: str = "uniform",
        weight: DomainFunction | ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
        data_accuracy_eps: float = 1e-12,
    ):
        if not isinstance(component.domain, RaggedSeriesDatasetDomain):
            raise TypeError(
                "RaggedSeriesSupervisedConstraint requires a "
                "RaggedSeriesDatasetDomain component."
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
        sampler_str = str(sampler)
        if sampler_str != "uniform":
            raise ValueError(
                "RaggedSeriesSupervisedConstraint supports only uniform sampling."
            )

        domain = component.domain
        self.constraint_vars = (str(constraint_var),)
        self.component = component
        self.structure = structure or ProductStructure((domain.labels,))
        self.dense_structure = None
        self.num_points = n
        self.sampler = sampler_str
        self.over = None
        self.reduction = reduction_value
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
    def domain(self) -> RaggedSeriesDatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, RaggedSeriesDatasetDomain):
            raise TypeError(
                "RaggedSeriesSupervisedConstraint domain is not a "
                "RaggedSeriesDatasetDomain."
            )
        return domain

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> RaggedSeriesSupervisedBatch:
        domain = self.domain
        indices = sample_case_indices(
            size=domain.size,
            num_samples=int(self.num_points),
            key=key,
            indices=self.indices,
        )
        points = domain.points_from_indices(indices, structure=self.structure)
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
        var = self.constraint_vars[0]
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
        batch: RaggedSeriesSupervisedBatch | None = None,
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
    "RaggedSeriesSupervisedBatch",
    "RaggedSeriesSupervisedConstraint",
]
