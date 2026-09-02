#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint
from .._likelihoods import AbstractLikelihood, GaussianLikelihood
from .._strict import StrictModule
from ..nn.operator.data import (
    FunctionSamples,
    OperatorBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from ..nn.operator.training import OperatorBatchLoader
from ._minibatch_posterior import AbstractObservationFactor, LikelihoodBatch
from ._operator import (
    _broadcast_named,
    _output_mask,
    _physical_dims,
    _queries_equal,
    _select_prediction_field,
)
from ._posterior_terms import AbstractPosteriorTerm


class OperatorStochasticGeometryPlan(StrictModule):
    """Explicit finite expected-log objective for stochastic anchors/geometry."""

    target: Literal["expected_log_likelihood"] = eqx.field(static=True)
    population_size: int = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: Literal["expected_log_likelihood"],
        population_size: int,
        geometry_epoch: int = 0,
    ):
        if target != "expected_log_likelihood":
            raise ValueError(
                "Stochastic operator geometry defines only expected_log_likelihood."
            )
        population = int(population_size)
        epoch = int(geometry_epoch)
        if population <= 0 or epoch < 0:
            raise ValueError(
                "Geometry population must be positive and epoch nonnegative."
            )
        self.target = target
        self.population_size = population
        self.geometry_epoch = epoch


class OperatorFactorSamplingPlan(StrictModule):
    """Audited query selection design and optional stochastic geometry objective."""

    design: Literal[
        "complete",
        "iid_nonuniform_with_replacement",
        "unequal_without_replacement",
        "fixed_subset_approximation",
    ] = eqx.field(static=True)
    query_ids: Array | None
    sampling_probabilities: Array | None
    estimator_weights: Array | None
    geometry: OperatorStochasticGeometryPlan | None
    unbiased: bool = eqx.field(static=True)

    def __init__(
        self,
        design: Literal[
            "complete",
            "iid_nonuniform_with_replacement",
            "unequal_without_replacement",
            "fixed_subset_approximation",
        ] = "complete",
        /,
        *,
        query_ids: ArrayLike | None = None,
        sampling_probabilities: ArrayLike | None = None,
        estimator_weights: ArrayLike | None = None,
        geometry: OperatorStochasticGeometryPlan | None = None,
    ):
        if design not in (
            "complete",
            "iid_nonuniform_with_replacement",
            "unequal_without_replacement",
            "fixed_subset_approximation",
        ):
            raise ValueError("Unknown operator factor sampling design.")
        if geometry is not None and not isinstance(
            geometry, OperatorStochasticGeometryPlan
        ):
            raise TypeError("geometry must be OperatorStochasticGeometryPlan or None.")
        supplied = (
            query_ids is not None,
            sampling_probabilities is not None,
            estimator_weights is not None,
        )
        if design == "complete":
            if any(supplied):
                raise ValueError("Complete query designs do not accept sampling arrays.")
        elif not all(supplied):
            raise ValueError(
                "Sampled query designs require IDs, probabilities, and estimator weights."
            )
        ids = None if query_ids is None else jnp.asarray(query_ids)
        probabilities = (
            None
            if sampling_probabilities is None
            else jnp.asarray(sampling_probabilities, dtype=float)
        )
        weights = (
            None
            if estimator_weights is None
            else jnp.asarray(estimator_weights, dtype=float)
        )
        if ids is not None:
            if not jnp.issubdtype(ids.dtype, jnp.integer):
                raise TypeError("query_ids must have integer dtype.")
            if probabilities.shape != ids.shape or weights.shape != ids.shape:
                raise ValueError("Operator query sampling arrays must align.")
            if (
                bool(jnp.any(~jnp.isfinite(probabilities)))
                or bool(jnp.any(probabilities <= 0.0))
                or bool(jnp.any(~jnp.isfinite(weights)))
                or bool(jnp.any(weights <= 0.0))
            ):
                raise ValueError(
                    "Operator query probabilities/weights must be finite and positive."
                )
        self.design = design
        self.query_ids = ids
        self.sampling_probabilities = probabilities
        self.estimator_weights = weights
        self.geometry = geometry
        self.unbiased = design in (
            "complete",
            "iid_nonuniform_with_replacement",
            "unequal_without_replacement",
        )


class OperatorLikelihoodData(StrictModule):
    """One operator batch with aligned targets and finite-observation masks."""

    batch: OperatorBatch
    target: Array
    output_spec: OperatorOutputSpec
    observation_mask: Array
    query_ids: Array
    query_sampling_probabilities: Array
    query_estimator_weights: Array
    geometry_epoch: int = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)

    def __init__(
        self,
        batch: OperatorBatch,
        target: ArrayLike | cx.Field | OperatorPrediction,
        /,
        *,
        output_spec: OperatorOutputSpec,
        field_name: str,
        query_name: str,
        observation_mask: ArrayLike | cx.Field | None = None,
        query_ids: ArrayLike | None = None,
        query_sampling_probabilities: ArrayLike | None = None,
        query_estimator_weights: ArrayLike | None = None,
        geometry_epoch: int = 0,
    ):
        if not isinstance(batch, OperatorBatch):
            raise TypeError("batch must be an OperatorBatch.")
        if not isinstance(output_spec, OperatorOutputSpec):
            raise TypeError("output_spec must be an OperatorOutputSpec.")
        selected_query_name = str(query_name)
        selected_field_name = str(field_name)
        if not selected_query_name or not selected_field_name:
            raise ValueError("field_name and query_name must be non-empty.")
        query = batch.query(selected_query_name)
        expected_shape = batch.case_shape + query.sample_shape + output_spec.channel_shape
        physical_dims = _physical_dims(query, output_spec, batch.case_axes)
        target_array = _target_array(
            target,
            batch=batch,
            query=query,
            field_name=selected_field_name,
            output_spec=output_spec,
            expected_shape=expected_shape,
            physical_dims=physical_dims,
        )
        combined_mask = _output_mask(
            query, output_spec, batch.case_shape
        ) & _observation_mask(
            observation_mask,
            expected_shape=expected_shape,
            physical_dims=physical_dims,
            has_channels=output_spec.channels != "scalar",
        )
        count = _case_count(batch.case_shape)
        if bool(jnp.any(~jnp.any(combined_mask.reshape((count, -1)), axis=-1))):
            raise ValueError(
                "Every physical operator case must contain at least one observation."
            )
        if bool(jnp.any(~jnp.isfinite(target_array) & combined_mask)):
            raise ValueError("Observed operator targets must be finite.")
        ids, probabilities, weights = _query_design(
            expected_shape,
            combined_mask,
            query_ids=query_ids,
            sampling_probabilities=query_sampling_probabilities,
            estimator_weights=query_estimator_weights,
        )
        epoch = int(geometry_epoch)
        if epoch < 0:
            raise ValueError("geometry_epoch must be nonnegative.")
        self.batch = batch
        self.target = jnp.where(combined_mask, target_array, 0.0)
        self.output_spec = output_spec
        self.observation_mask = combined_mask
        self.query_ids = ids
        self.query_sampling_probabilities = probabilities
        self.query_estimator_weights = weights
        self.geometry_epoch = epoch
        self.case_count = count
        self.field_name = selected_field_name
        self.query_name = selected_query_name


class OperatorBatchObservationLikelihood(AbstractObservationFactor):
    """Per-case normalized likelihood for dynamically supplied operator batches."""

    likelihood: AbstractLikelihood
    predict_fn: Callable[[PyTree[Any], OperatorBatch], OperatorPrediction] = eqx.field(
        static=True
    )
    parameters_fn: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None = (
        eqx.field(static=True)
    )
    label: str = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)
    semantics: str = eqx.field(static=True)

    def __init__(
        self,
        predict: Callable[[PyTree[Any], OperatorBatch], OperatorPrediction],
        likelihood: AbstractLikelihood,
        /,
        *,
        parameters: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]]
        | None = None,
        label: str = "operator_observation",
        factor_id: str = "operator-observation",
    ):
        if not callable(predict):
            raise TypeError("predict must be callable.")
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if parameters is not None and not callable(parameters):
            raise TypeError("parameters must be callable or None.")
        self.likelihood = likelihood
        self.predict_fn = predict
        self.parameters_fn = parameters
        self.label = _label(label)
        selected_factor_id = str(factor_id)
        if not selected_factor_id:
            raise ValueError("factor_id must be non-empty.")
        self.factor_id = selected_factor_id
        self.semantics = "normalized_likelihood"

    def _likelihood_parameters(self, parameters: PyTree[Any], /) -> dict[str, Array]:
        return _likelihood_parameter_values(self.parameters_fn, parameters)

    def per_case_log_prob(
        self,
        parameters: PyTree[Any],
        data: OperatorLikelihoodData,
        /,
    ) -> Array:
        if not isinstance(data, OperatorLikelihoodData):
            raise TypeError("data must be OperatorLikelihoodData.")
        prediction = _validated_operator_prediction(
            self.predict_fn(parameters, data.batch),
            batch=data.batch,
            target=data.target,
            output_spec=data.output_spec,
            field_name=data.field_name,
            query_name=data.query_name,
        )
        return _operator_per_case_log_prob(
            prediction,
            target=data.target,
            observation_mask=data.observation_mask,
            query_estimator_weights=data.query_estimator_weights,
            likelihood=self.likelihood,
            likelihood_parameters=self._likelihood_parameters(parameters),
            case_count=data.case_count,
        )

    def log_factors(
        self,
        parameters: PyTree[Any],
        batch: LikelihoodBatch,
        /,
    ) -> Array:
        if not isinstance(batch, LikelihoodBatch):
            raise TypeError("batch must be a LikelihoodBatch.")
        factors = self.per_case_log_prob(parameters, batch.data)
        if factors.shape != batch.factor_mask.shape:
            raise ValueError(
                "Operator likelihood factors and batch factor_mask must align."
            )
        return jnp.where(batch.factor_mask, factors, jnp.zeros_like(factors))


class OperatorMinibatchSource:
    """Padded deterministic likelihood batches adapted from operator training data."""

    def __init__(
        self,
        loader: OperatorBatchLoader,
        /,
        *,
        field_name: str,
        observation_mask: ArrayLike | None = None,
        factor_sampling: OperatorFactorSamplingPlan | None = None,
    ):
        if not isinstance(loader, OperatorBatchLoader):
            raise TypeError("loader must be an OperatorBatchLoader.")
        if loader.drop_last:
            raise ValueError("Operator SG-MCMC does not permit drop_last=True.")
        if not loader.shuffle:
            raise ValueError("Operator SG-MCMC requires shuffle=True.")
        sampling_plan = (
            OperatorFactorSamplingPlan() if factor_sampling is None else factor_sampling
        )
        if not isinstance(sampling_plan, OperatorFactorSamplingPlan):
            raise TypeError("factor_sampling must be OperatorFactorSamplingPlan or None.")
        if loader.sampling is not None and sampling_plan.geometry is None:
            raise ValueError(
                "Mutating operator anchors/geometry require an explicit "
                "expected_log_likelihood geometry plan."
            )
        if jax.process_count() != 1:
            raise ValueError("Operator SG-MCMC currently requires one JAX process.")
        selected_field_name = str(field_name)
        if not selected_field_name:
            raise ValueError("field_name must be non-empty.")
        first = loader.prepare_indices((0,), epoch=0, batch_index=0)
        field = first.targets.field(selected_field_name)
        mask = (
            None
            if observation_mask is None
            else jnp.asarray(observation_mask, dtype=bool)
        )
        if mask is not None and (
            mask.ndim == 0 or int(mask.shape[0]) != loader.source.size
        ):
            raise ValueError("observation_mask must have the source case axis in front.")
        loader_fingerprint = loader.fingerprint
        configuration = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "loader": loader.configuration(),
            "loader_fingerprint": loader_fingerprint,
            "field_name": selected_field_name,
            "query_name": field.query_name,
            "output_spec": {
                "channels": field.spec.channels,
                "component_names": list(field.spec.component_names),
            },
            "observation_mask": (None if mask is None else array_tree_fingerprint(mask)),
            "factor_sampling": {
                "design": sampling_plan.design,
                "unbiased": sampling_plan.unbiased,
                "query_design": (
                    None
                    if sampling_plan.query_ids is None
                    else array_tree_fingerprint(
                        {
                            "ids": sampling_plan.query_ids,
                            "probabilities": sampling_plan.sampling_probabilities,
                            "weights": sampling_plan.estimator_weights,
                        }
                    )
                ),
                "geometry": (
                    None
                    if sampling_plan.geometry is None
                    else {
                        "target": sampling_plan.geometry.target,
                        "population_size": sampling_plan.geometry.population_size,
                        "geometry_epoch": sampling_plan.geometry.geometry_epoch,
                    }
                ),
            },
        }
        configuration_json = json.dumps(
            configuration,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.loader = loader
        self.field_name = selected_field_name
        self.factor_sampling = sampling_plan
        self.query_name = field.query_name
        self.output_spec = field.spec
        self.observation_mask = mask
        self._configuration_json = configuration_json
        self._fingerprint = hashlib.sha256(configuration_json.encode("utf-8")).hexdigest()

    @property
    def num_factors(self) -> int:
        return int(self.loader.source.size)

    @property
    def batch_capacity(self) -> int:
        return self.loader.batch_size

    @property
    def batches_per_epoch(self) -> int:
        return self.loader.batches_per_epoch

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def configuration(self) -> Mapping[str, Any]:
        return json.loads(self._configuration_json)

    def epoch(self, epoch: int, /) -> Iterator[LikelihoodBatch]:
        epoch_index = int(epoch)
        if epoch_index < 0:
            raise ValueError("epoch must be nonnegative.")
        with self.loader.epoch(epoch_index) as batches:
            for prepared in batches:
                yield self._likelihood_batch(prepared, audit=False)

    def audit_epoch(self) -> Iterator[LikelihoodBatch]:
        """Enumerate all physical cases exactly once with unit weights."""
        for batch_index, start in enumerate(
            range(0, self.num_factors, self.batch_capacity)
        ):
            indices = tuple(
                range(start, min(start + self.batch_capacity, self.num_factors))
            )
            prepared = self.loader.prepare_indices(
                indices, epoch=0, batch_index=batch_index
            )
            yield self._likelihood_batch(prepared, audit=True)

    def _likelihood_batch(self, prepared: Any, /, *, audit: bool) -> LikelihoodBatch:
        indices = prepared.indices
        active_count = len(indices)
        padded_indices = indices + (indices[-1],) * (self.batch_capacity - active_count)
        operator_batch = prepared.batch
        operator_targets = prepared.targets
        if active_count < self.batch_capacity:
            selection = jnp.concatenate(
                (
                    jnp.arange(active_count),
                    jnp.full(
                        (self.batch_capacity - active_count,),
                        active_count - 1,
                    ),
                )
            )
            operator_batch = operator_batch.take(selection)
            operator_targets = operator_targets.take(selection)
        target_field = operator_targets.field(self.field_name)
        selected_mask = (
            None
            if self.observation_mask is None
            else self.observation_mask[jnp.asarray(padded_indices)]
        )
        data = OperatorLikelihoodData(
            operator_batch,
            target_field.values,
            output_spec=self.output_spec,
            field_name=self.field_name,
            query_name=self.query_name,
            observation_mask=selected_mask,
            query_ids=self.factor_sampling.query_ids,
            query_sampling_probabilities=(self.factor_sampling.sampling_probabilities),
            query_estimator_weights=self.factor_sampling.estimator_weights,
            geometry_epoch=(
                0
                if self.factor_sampling.geometry is None
                else self.factor_sampling.geometry.geometry_epoch
            ),
        )
        factor_mask = jnp.arange(self.batch_capacity) < active_count
        batch_indices = jnp.asarray(padded_indices, dtype=jnp.int32)
        return LikelihoodBatch(
            data,
            factor_mask,
            factor_ids=batch_indices,
            sampling_probabilities=jnp.full(
                (self.batch_capacity,), 1.0 / self.num_factors
            ),
            estimator_weights=jnp.full(
                (self.batch_capacity,),
                1.0 if audit else self.num_factors / active_count,
            ),
        )


class FixedOperatorObservationLikelihood(AbstractPosteriorTerm):
    """Normalized finite-observation likelihood for one fixed operator batch."""

    batch: OperatorBatch
    target: Array
    likelihood: AbstractLikelihood
    output_spec: OperatorOutputSpec
    observation_mask: Array
    query_sampling_probabilities: Array
    query_estimator_weights: Array
    geometry_epoch: int = eqx.field(static=True)
    predict_fn: Callable[[PyTree[Any]], OperatorPrediction] = eqx.field(static=True)
    parameters_fn: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None = (
        eqx.field(static=True)
    )
    case_count: int = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)

    def __init__(
        self,
        predict: Callable[[PyTree[Any]], OperatorPrediction],
        batch: OperatorBatch,
        target: ArrayLike | cx.Field | OperatorPrediction,
        likelihood: AbstractLikelihood,
        /,
        *,
        output_spec: OperatorOutputSpec,
        field_name: str,
        query_name: str,
        observation_mask: ArrayLike | cx.Field | None = None,
        query_sampling_probabilities: ArrayLike | None = None,
        query_estimator_weights: ArrayLike | None = None,
        geometry_epoch: int = 0,
        parameters: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]]
        | None = None,
        label: str = "operator_observation",
    ):
        if not callable(predict):
            raise TypeError("predict must be callable.")
        if not isinstance(batch, OperatorBatch):
            raise TypeError("batch must be an OperatorBatch.")
        if not isinstance(output_spec, OperatorOutputSpec):
            raise TypeError("output_spec must be an OperatorOutputSpec.")
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if parameters is not None and not callable(parameters):
            raise TypeError("parameters must be callable or None.")

        selected_query_name = str(query_name)
        selected_field_name = str(field_name)
        if not selected_query_name or not selected_field_name:
            raise ValueError("field_name and query_name must be non-empty.")
        query = batch.query(selected_query_name)
        expected_shape = batch.case_shape + query.sample_shape + output_spec.channel_shape
        physical_dims = _physical_dims(query, output_spec, batch.case_axes)
        target_array = _target_array(
            target,
            batch=batch,
            query=query,
            field_name=selected_field_name,
            output_spec=output_spec,
            expected_shape=expected_shape,
            physical_dims=physical_dims,
        )
        query_mask = _output_mask(query, output_spec, batch.case_shape)
        user_mask = _observation_mask(
            observation_mask,
            expected_shape=expected_shape,
            physical_dims=physical_dims,
            has_channels=output_spec.channels != "scalar",
        )
        combined_mask = query_mask & user_mask
        count = _case_count(batch.case_shape)
        per_case_mask = combined_mask.reshape((count, -1))
        if bool(jnp.any(~jnp.any(per_case_mask, axis=-1))):
            raise ValueError(
                "Every physical operator case must contain at least one observation."
            )
        if bool(jnp.any(~jnp.isfinite(target_array) & combined_mask)):
            raise ValueError("Observed operator targets must be finite.")
        _, query_probabilities, query_weights = _query_design(
            expected_shape,
            combined_mask,
            query_ids=None,
            sampling_probabilities=query_sampling_probabilities,
            estimator_weights=query_estimator_weights,
        )
        epoch = int(geometry_epoch)
        if epoch < 0:
            raise ValueError("geometry_epoch must be nonnegative.")

        self.batch = batch
        self.target = jnp.where(combined_mask, target_array, 0.0)
        self.likelihood = likelihood
        self.output_spec = output_spec
        self.observation_mask = combined_mask
        self.query_sampling_probabilities = query_probabilities
        self.query_estimator_weights = query_weights
        self.geometry_epoch = epoch
        self.predict_fn = predict
        self.parameters_fn = parameters
        self.case_count = count
        self.label = _label(label)
        self.field_name = selected_field_name
        self.query_name = selected_query_name

    def _prediction(self, parameters: PyTree[Any], /) -> Array:
        return _validated_operator_prediction(
            self.predict_fn(parameters),
            batch=self.batch,
            target=self.target,
            output_spec=self.output_spec,
            field_name=self.field_name,
            query_name=self.query_name,
            contract="fixed batch",
        )

    def _likelihood_parameters(
        self,
        parameters: PyTree[Any],
        /,
    ) -> dict[str, Array]:
        return _likelihood_parameter_values(self.parameters_fn, parameters)

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        return _operator_per_case_log_prob(
            self._prediction(parameters),
            target=self.target,
            observation_mask=self.observation_mask,
            query_estimator_weights=self.query_estimator_weights,
            likelihood=self.likelihood,
            likelihood_parameters=self._likelihood_parameters(parameters),
            case_count=self.case_count,
        )

    def standardized_residual(self, parameters: PyTree[Any], /) -> Array:
        """Return fixed-Gaussian residuals with masked components set to zero."""
        if not isinstance(self.likelihood, GaussianLikelihood):
            raise TypeError("standardized_residual requires a fixed GaussianLikelihood.")
        prediction = self._prediction(parameters)
        scale = jnp.broadcast_to(self.likelihood.scale, self.target.shape)
        residual = (self.target - prediction) / scale
        return jnp.where(self.observation_mask, residual, 0.0)


def _validated_operator_prediction(
    prediction: OperatorPrediction,
    /,
    *,
    batch: OperatorBatch,
    target: Array,
    output_spec: OperatorOutputSpec,
    field_name: str,
    query_name: str,
    contract: str = "batch",
) -> Array:
    if not isinstance(prediction, OperatorPrediction):
        raise TypeError("Operator likelihood prediction must be OperatorPrediction.")
    _, field, query = _select_prediction_field(prediction, field_name)
    if (
        prediction.case_axes != batch.case_axes
        or prediction.case_shape != batch.case_shape
        or field.spec.channels != output_spec.channels
        or field.spec.component_names != output_spec.component_names
    ):
        raise ValueError(
            f"Operator likelihood prediction does not match the {contract} contract."
        )
    values = jnp.asarray(field.values)
    if values.shape != target.shape:
        raise ValueError(
            f"Operator likelihood prediction must have shape {target.shape}; "
            f"got {values.shape}."
        )
    return _checked_query_values(
        values,
        query,
        batch.query(query_name),
    )


def _likelihood_parameter_values(
    callback: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None,
    parameters: PyTree[Any],
    /,
) -> dict[str, Array]:
    if callback is None:
        return {}
    values = callback(parameters)
    if not isinstance(values, Mapping):
        raise TypeError("Likelihood parameters callback must return a mapping.")
    return {
        str(name): jnp.asarray(value.data if isinstance(value, cx.Field) else value)
        for name, value in values.items()
    }


def _operator_per_case_log_prob(
    prediction: Array,
    /,
    *,
    target: Array,
    observation_mask: Array,
    likelihood: AbstractLikelihood,
    query_estimator_weights: Array,
    likelihood_parameters: Mapping[str, Array],
    case_count: int,
) -> Array:
    safe_prediction = jnp.where(observation_mask, prediction, 0.0)
    values = jnp.asarray(
        likelihood.log_prob(
            safe_prediction,
            target,
            **likelihood_parameters,
        )
    )
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("Operator observation log densities must be real floating.")
    values = jnp.broadcast_to(values, target.shape)
    invalid_prediction = observation_mask & ~jnp.isfinite(prediction)
    invalid_density = observation_mask & ~jnp.isfinite(values)
    elements = jnp.where(observation_mask, query_estimator_weights * values, 0.0)
    elements = jnp.where(
        invalid_prediction | invalid_density,
        -jnp.inf,
        elements,
    )
    return jnp.sum(elements.reshape((case_count, -1)), axis=-1)


def _query_design(
    expected_shape: tuple[int, ...],
    observation_mask: Array,
    /,
    *,
    query_ids: ArrayLike | None,
    sampling_probabilities: ArrayLike | None,
    estimator_weights: ArrayLike | None,
) -> tuple[Array, Array, Array]:
    if (sampling_probabilities is None) != (estimator_weights is None):
        raise ValueError(
            "Query sampling probabilities and estimator weights must be supplied together."
        )
    ids = (
        jnp.arange(math.prod(expected_shape), dtype=jnp.int32).reshape(expected_shape)
        if query_ids is None
        else jnp.broadcast_to(jnp.asarray(query_ids), expected_shape)
    )
    if not jnp.issubdtype(ids.dtype, jnp.integer):
        raise TypeError("query_ids must be an integer array broadcastable to targets.")
    probabilities = (
        jnp.ones(expected_shape, dtype=float)
        if sampling_probabilities is None
        else jnp.broadcast_to(
            jnp.asarray(sampling_probabilities, dtype=float), expected_shape
        )
    )
    weights = (
        jnp.ones(expected_shape, dtype=float)
        if estimator_weights is None
        else jnp.broadcast_to(jnp.asarray(estimator_weights, dtype=float), expected_shape)
    )
    if bool(
        jnp.any(
            (
                ~jnp.isfinite(probabilities)
                | ~jnp.isfinite(weights)
                | (probabilities <= 0.0)
                | (weights <= 0.0)
            )
            & observation_mask
        )
    ):
        raise ValueError(
            "Active query probabilities and estimator weights must be finite and positive."
        )
    return (
        jnp.where(observation_mask, ids, -jnp.ones_like(ids)),
        jnp.where(observation_mask, probabilities, jnp.ones_like(probabilities)),
        jnp.where(observation_mask, weights, jnp.zeros_like(weights)),
    )


def _target_array(
    target: ArrayLike | cx.Field | OperatorPrediction,
    /,
    *,
    batch: OperatorBatch,
    query: FunctionSamples,
    field_name: str,
    output_spec: OperatorOutputSpec,
    expected_shape: tuple[int, ...],
    physical_dims: tuple[str, ...],
) -> Array:
    if isinstance(target, OperatorPrediction):
        _, field, target_query = _select_prediction_field(target, field_name)
        if (
            target.case_axes != batch.case_axes
            or target.case_shape != batch.case_shape
            or field.spec.channels != output_spec.channels
            or field.spec.component_names != output_spec.component_names
            or not _queries_equal(target_query, query)
        ):
            raise ValueError("Operator target does not match the fixed batch contract.")
        target_array = jnp.asarray(field.values)
    elif isinstance(target, cx.Field):
        template = cx.Field(jnp.empty(expected_shape), dims=physical_dims)
        target_array = jnp.asarray(_broadcast_named(target, template))
    else:
        target_array = jnp.asarray(target)
    if target_array.shape != expected_shape:
        raise ValueError(
            f"Operator target must have shape {expected_shape}; got {target_array.shape}."
        )
    return target_array


def _checked_query_values(
    values: Array,
    left: FunctionSamples,
    right: FunctionSamples,
    /,
) -> Array:
    if len(left.axes) != len(right.axes):
        raise ValueError(
            "Operator likelihood prediction does not match the fixed batch contract."
        )
    equal = jnp.asarray(True)
    for left_axis, right_axis in zip(left.axes, right.axes, strict=True):
        if (
            left_axis.name != right_axis.name
            or left_axis.basis != right_axis.basis
            or left_axis.periodic != right_axis.periodic
            or left_axis.nodes.shape != right_axis.nodes.shape
            or not _same_optional_structure(
                left_axis.quadrature_weights,
                right_axis.quadrature_weights,
            )
        ):
            raise ValueError(
                "Operator likelihood prediction does not match the fixed batch contract."
            )
        equal = equal & jnp.array_equal(left_axis.nodes, right_axis.nodes)
        left_weights = left_axis.quadrature_weights
        right_weights = right_axis.quadrature_weights
        if left_weights is not None and right_weights is not None:
            equal = equal & jnp.array_equal(left_weights, right_weights)
    for left_value, right_value in (
        (left.coordinates, right.coordinates),
        (left.quadrature_weights, right.quadrature_weights),
        (left.mask, right.mask),
    ):
        if not _same_optional_structure(left_value, right_value):
            raise ValueError(
                "Operator likelihood prediction does not match the fixed batch contract."
            )
        if left_value is not None and right_value is not None:
            equal = equal & jnp.array_equal(left_value, right_value)
    return eqx.error_if(
        values,
        ~equal,
        "Operator likelihood prediction does not match the fixed batch contract.",
    )


def _same_optional_structure(
    left: Array | None,
    right: Array | None,
    /,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.shape == right.shape


def _observation_mask(
    mask: ArrayLike | cx.Field | None,
    /,
    *,
    expected_shape: tuple[int, ...],
    physical_dims: tuple[str, ...],
    has_channels: bool,
) -> Array:
    if mask is None:
        return jnp.ones(expected_shape, dtype=bool)
    if isinstance(mask, cx.Field):
        template = cx.Field(jnp.empty(expected_shape), dims=physical_dims)
        return jnp.asarray(_broadcast_named(mask, template), dtype=bool)
    value = jnp.asarray(mask, dtype=bool)
    if has_channels and value.shape == expected_shape[:-1]:
        value = value[..., None]
    return jnp.broadcast_to(value, expected_shape)


def _case_count(case_shape: tuple[int, ...], /) -> int:
    count = 1
    for size in case_shape:
        count *= int(size)
    return count


def _label(value: str, /) -> str:
    label = str(value)
    if not label:
        raise ValueError("label must be non-empty.")
    return label


__all__ = [
    "FixedOperatorObservationLikelihood",
    "OperatorBatchObservationLikelihood",
    "OperatorFactorSamplingPlan",
    "OperatorLikelihoodData",
    "OperatorMinibatchSource",
    "OperatorStochasticGeometryPlan",
]
