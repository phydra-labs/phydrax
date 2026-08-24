#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DatasetDomain, DomainComponent, DomainFunction, PointSampling

from .._doc import DOC_KEY0
from .._likelihoods import AbstractLikelihood
from .._term import AbstractSamplingTerm
from ._data_metrics import (
    case_sample_count,
    configured_case_indices,
    normalize_case_sampling,
    reduce_supervised_loss,
    sample_case_indices,
    validate_case_weights,
    validate_supervised_targets,
)
from ._supervised_dataset import SupervisedDatasetBatch


class _AbstractSupervisedLikelihoodTerm(AbstractSamplingTerm):
    """Shared dataset-likelihood sampling and reduction implementation."""

    __strict_abstract__ = True

    fields: tuple[str, ...]
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
    sample_weight: Array | None
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
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        owner = type(self).__name__
        if not isinstance(component.domain, DatasetDomain):
            raise TypeError(f"{owner} requires a DatasetDomain component.")
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        sampling_ = normalize_case_sampling(
            sampling,
            labels=component.domain.labels,
            owner=owner,
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
        self.fields = variables
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
        configured = configured_case_indices(
            indices,
            sample_mask,
            size=domain.size,
        )
        self.sample_weight = validate_case_weights(
            sample_weight,
            size=domain.size,
            indices=configured,
        )
        self.indices = configured
        self.label = None if label is None else str(label)

    @property
    def domain(self) -> DatasetDomain:
        domain = self.component.domain
        if not isinstance(domain, DatasetDomain):
            raise TypeError("Likelihood term domain is not a DatasetDomain.")
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
            sample_weight=(
                None if self.sample_weight is None else self.sample_weight[indices]
            ),
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
            sample_weight=(
                None if self.sample_weight is None else self.sample_weight[indices]
            ),
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
        location, target = self.likelihood.align_observations(
            self._location(functions, batch, key=key, **kwargs),
            batch.target,
        )
        parameters: dict[str, Array] = {}
        if self.scale_var is not None:
            raw_scale_field = functions[self.scale_var](batch.points, key=key, **kwargs)
            if not isinstance(raw_scale_field, cx.Field):
                raise TypeError("Likelihood scale must evaluate to a coordax.Field.")
            raw_scale, _ = self.likelihood.align_observations(
                jnp.asarray(raw_scale_field.data), target
            )
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
        iter_: int | Array | None = None,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        batch_value = self.sample(key=key) if batch is None else batch

        def zero_loss() -> Array:
            return jnp.zeros((), dtype=jnp.result_type(self.weight, float))

        def active_loss() -> Array:
            per_case = -self.log_prob(
                functions,
                key=key,
                batch=batch_value,
                **kwargs,
            )
            reduced = reduce_supervised_loss(
                per_case,
                reduction=self.reduction,
                sample_weight=batch_value.sample_weight,
            )
            return self.weight * jnp.asarray(reduced, dtype=float).reshape(())

        return jax.lax.cond(self.weight == 0.0, zero_loss, active_loss)


class SupervisedLikelihoodTerm(_AbstractSupervisedLikelihoodTerm):
    """Score direct or operator-transformed dataset observations by a likelihood."""


__all__ = ["SupervisedLikelihoodTerm"]
