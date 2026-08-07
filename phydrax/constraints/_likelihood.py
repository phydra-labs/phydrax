#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DatasetDomain, DomainComponent, DomainFunction, PointSampling

from .._doc import DOC_KEY0
from ..uq._likelihoods import AbstractLikelihood
from ._base import AbstractSamplingConstraint
from ._data_metrics import (
    case_sample_count,
    normalize_case_sampling,
    sample_case_indices,
    validate_case_indices,
    validate_supervised_targets,
)
from ._supervised_dataset import SupervisedDatasetBatch


class SupervisedLikelihoodConstraint(AbstractSamplingConstraint):
    """Score direct or operator-transformed dataset observations by a likelihood."""

    constraint_vars: tuple[str, ...]
    location_var: str
    scale_var: str | None
    component: DomainComponent
    sampling: PointSampling
    over: None
    reduction: Literal["mean", "sum"]
    values: Array
    likelihood: AbstractLikelihood
    observation_operator: Callable[[DomainFunction], DomainFunction] | None
    weight: Array
    indices: Array | None
    label: str | None

    def __init__(
        self,
        location_var: str,
        component: DomainComponent,
        values: ArrayLike,
        likelihood: AbstractLikelihood,
        /,
        *,
        sampling: PointSampling,
        scale_var: str | None = None,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        if not isinstance(component.domain, DatasetDomain):
            raise TypeError(
                "SupervisedLikelihoodConstraint requires a DatasetDomain component."
            )
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        sampling_ = normalize_case_sampling(
            sampling,
            labels=component.domain.labels,
            owner="SupervisedLikelihoodConstraint",
        )
        reduction_value = str(reduction)
        if reduction_value not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        reduction_: Literal["mean", "sum"] = (
            "mean" if reduction_value == "mean" else "sum"
        )
        location_name = str(location_var)
        scale_name = None if scale_var is None else str(scale_var)
        variables = (
            (location_name,) if scale_name is None else (location_name, scale_name)
        )
        domain = component.domain
        self.constraint_vars = variables
        self.location_var = location_name
        self.scale_var = scale_name
        self.component = component
        self.sampling = sampling_
        self.over = None
        self.reduction = reduction_
        self.values = validate_supervised_targets(
            values,
            leading_size=domain.size,
            name="supervised likelihood target",
        )
        self.likelihood = likelihood
        self.observation_operator = observation_operator
        weight_array = jnp.asarray(weight, dtype=float)
        if weight_array.ndim != 0 or not bool(jnp.isfinite(weight_array)):
            raise ValueError("weight must be a finite scalar.")
        self.weight = weight_array
        self.indices = validate_case_indices(indices, size=domain.size, name="indices")
        self.label = None if label is None else str(label)

    @property
    def domain(self) -> DatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, DatasetDomain):
            raise TypeError("Likelihood constraint domain is not a DatasetDomain.")
        return domain


    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> Any:
        indices = sample_case_indices(
            size=self.domain.size,
            num_samples=case_sample_count(self.sampling),
            key=key,
            indices=self.indices,
        )
        layout = self.sampling.layout
        assert layout is not None
        return SupervisedDatasetBatch(
            points=self.domain.points_from_indices(indices, structure=layout),
            target=self.values[indices],
            indices=indices,
        )

    def observed_batch(self) -> SupervisedDatasetBatch:
        """Return every configured observation exactly once without random sampling."""
        indices = (
            jnp.arange(self.domain.size, dtype=jnp.int32)
            if self.indices is None
            else self.indices
        )
        layout = self.sampling.layout
        assert layout is not None
        return SupervisedDatasetBatch(
            points=self.domain.points_from_indices(indices, structure=layout),
            target=self.values[indices],
            indices=indices,
        )

    def log_prob(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch,
        **kwargs: Any,
    ) -> Array:
        """Return unreduced per-case observation log probabilities for a fixed batch."""
        location, target = _align_observations(
            self._location(functions, batch, key=key, **kwargs),
            batch.target,
        )
        parameters: dict[str, Array] = {}
        if self.scale_var is not None:
            raw_scale_field = functions[self.scale_var](batch.points, key=key, **kwargs)
            if not isinstance(raw_scale_field, cx.Field):
                raise TypeError("Likelihood scale must evaluate to a coordax.Field.")
            raw_scale, _ = _align_observations(jnp.asarray(raw_scale_field.data), target)
            parameters["raw_scale"] = raw_scale
        log_prob = jnp.asarray(
            self.likelihood.log_prob(location, target, **parameters), dtype=float
        )
        if log_prob.ndim == 0 or int(log_prob.shape[0]) != int(target.shape[0]):
            raise ValueError(
                "Likelihood log_prob must retain the leading empirical-case axis."
            )
        return log_prob.reshape((int(log_prob.shape[0]), -1)).sum(axis=1)

    def _location(
        self,
        functions: Mapping[str, DomainFunction],
        batch: SupervisedDatasetBatch,
        /,
        *,
        key: Key[Array, ""],
        **kwargs: Any,
    ) -> Array:
        function = functions[self.location_var]
        if self.observation_operator is not None:
            function = self.observation_operator(function)
            if not isinstance(function, DomainFunction):
                raise TypeError("observation_operator must return a DomainFunction.")
        value = function(batch.points, key=key, **kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("Likelihood location must evaluate to a coordax.Field.")
        return jnp.asarray(value.data, dtype=float)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        batch_value = self.sample(key=key) if batch is None else batch
        per_case = -self.log_prob(
            functions,
            key=key,
            batch=batch_value,
            **kwargs,
        )
        if self.reduction == "mean":
            reduced = jnp.mean(per_case)
        else:
            reduced = jnp.sum(per_case)
        return self.weight * jnp.asarray(reduced, dtype=float).reshape(())


def _align_observations(prediction: ArrayLike, target: ArrayLike) -> tuple[Array, Array]:
    prediction_array = jnp.asarray(prediction, dtype=float)
    target_array = jnp.asarray(target, dtype=float)
    if prediction_array.shape == target_array.shape:
        return prediction_array, target_array
    if (
        prediction_array.ndim == 2
        and target_array.ndim == 1
        and prediction_array.shape[1] == 1
    ):
        prediction_array = prediction_array[:, 0]
    elif (
        target_array.ndim == 2
        and prediction_array.ndim == 1
        and target_array.shape[1] == 1
    ):
        target_array = target_array[:, 0]
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "Likelihood prediction and target shapes are incompatible: "
            f"prediction={prediction_array.shape}, target={target_array.shape}."
        )
    return prediction_array, target_array


__all__ = ["SupervisedLikelihoodConstraint"]
