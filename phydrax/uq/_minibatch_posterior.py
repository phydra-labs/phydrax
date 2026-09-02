#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PyTree

from .._data_plane import EPOCH_ORDER_ALGORITHM, IndexEpochPlan
from .._fingerprint import array_tree_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ._posterior import ParameterSpace


ObservationFactorSemantics = Literal["normalized_likelihood", "unnormalized_potential"]


class AbstractObservationFactor(StrictModule):
    """Immutable fixed-capacity observation-factor contract."""

    factor_id: str = eqx.field(static=True)
    semantics: ObservationFactorSemantics = eqx.field(static=True)

    @abstractmethod
    def log_factors(self, physical: PyTree[Any], batch: LikelihoodBatch, /) -> Array:
        """Return one real log factor per batch slot."""
        raise NotImplementedError


class FactorSamplingState(StrictModule):
    """Source progress recorded independently from generic chain state."""

    epoch: int = eqx.field(static=True)
    batch_index: int = eqx.field(static=True)
    probability_epoch: int = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)

    def __init__(
        self,
        epoch: int = 0,
        batch_index: int = 0,
        probability_epoch: int = 0,
        geometry_epoch: int = 0,
    ):
        values = tuple(
            int(value)
            for value in (epoch, batch_index, probability_epoch, geometry_epoch)
        )
        if any(value < 0 for value in values):
            raise ValueError("Factor sampling state indices must be nonnegative.")
        (
            self.epoch,
            self.batch_index,
            self.probability_epoch,
            self.geometry_epoch,
        ) = values


class LikelihoodBatch(StrictModule):
    """Fixed-capacity likelihood data and audited estimator design."""

    data: PyTree[Any]
    factor_mask: Array
    factor_ids: Array
    sampling_probabilities: Array
    estimator_weights: Array

    def __init__(
        self,
        data: PyTree[Any],
        factor_mask: ArrayLike,
        /,
        *,
        factor_ids: ArrayLike,
        sampling_probabilities: ArrayLike,
        estimator_weights: ArrayLike,
    ):
        mask = jnp.asarray(factor_mask)
        if mask.ndim != 1:
            raise ValueError("factor_mask must be one-dimensional.")
        if mask.dtype != jnp.bool_:
            raise TypeError("factor_mask must have boolean dtype.")
        if mask.shape[0] == 0:
            raise ValueError("factor_mask must have positive capacity.")
        if not bool(jnp.any(mask)):
            raise ValueError("LikelihoodBatch must contain at least one active factor.")
        ids = jnp.asarray(factor_ids)
        probabilities = jnp.asarray(sampling_probabilities)
        weights = jnp.asarray(estimator_weights)
        if ids.shape != mask.shape or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError("factor_ids must be an integer array aligned to factor_mask.")
        for name, values in (
            ("sampling_probabilities", probabilities),
            ("estimator_weights", weights),
        ):
            if values.shape != mask.shape or not jnp.issubdtype(
                values.dtype, jnp.floating
            ):
                raise TypeError(
                    f"{name} must be a real floating array aligned to factor_mask."
                )
            if bool(jnp.any(~jnp.isfinite(values) & mask)):
                raise ValueError(f"Active {name} values must be finite.")
        if bool(jnp.any((ids < 0) & mask)):
            raise ValueError("Active factor_ids must be nonnegative.")
        if bool(jnp.any((probabilities <= 0.0) & mask)):
            raise ValueError("Active sampling probabilities must be strictly positive.")
        if bool(jnp.any((weights <= 0.0) & mask)):
            raise ValueError("Active estimator weights must be strictly positive.")
        self.data = data
        self.factor_mask = mask
        self.factor_ids = jnp.where(mask, ids, -jnp.ones_like(ids))
        self.sampling_probabilities = jnp.where(
            mask, probabilities, jnp.ones_like(probabilities)
        )
        self.estimator_weights = jnp.where(mask, weights, jnp.zeros_like(weights))

    @property
    def capacity(self) -> int:
        return int(self.factor_mask.shape[0])

    @property
    def factor_count(self) -> Array:
        return jnp.sum(self.factor_mask, dtype=jnp.int32)


@runtime_checkable
class MinibatchSource(Protocol):
    """Deterministic finite source with sampled and exact-audit epochs."""

    @property
    def num_factors(self) -> int: ...

    @property
    def batch_capacity(self) -> int: ...

    @property
    def batches_per_epoch(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def configuration(self) -> Mapping[str, Any]: ...

    def epoch(self, epoch: int, /) -> Iterator[LikelihoodBatch]: ...

    def audit_epoch(self) -> Iterator[LikelihoodBatch]: ...


class ArrayMinibatchSource(StrictModule):
    """Deterministic shuffled epochs over array PyTrees with padded remainders."""

    data: PyTree[Array]
    _num_factors: int = eqx.field(static=True)
    _batch_capacity: int = eqx.field(static=True)
    _seed: int = eqx.field(static=True)
    _batches_per_epoch: int = eqx.field(static=True)
    _configuration_json: str = eqx.field(static=True)
    _fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        data: PyTree[ArrayLike],
        /,
        *,
        batch_size: int,
        seed: int = 0,
    ):
        leaves = jax.tree_util.tree_leaves(data)
        if not leaves:
            raise ValueError("ArrayMinibatchSource data must contain array leaves.")
        arrays = jax.tree_util.tree_map(jnp.asarray, data)
        array_leaves = jax.tree_util.tree_leaves(arrays)
        population = int(array_leaves[0].shape[0]) if array_leaves[0].ndim else 0
        if population <= 0:
            raise ValueError("Every source data leaf needs a positive leading axis.")
        for leaf in array_leaves:
            if leaf.ndim == 0 or int(leaf.shape[0]) != population:
                raise ValueError(
                    "Every source data leaf must share the positive factor-leading axis."
                )
        capacity = int(batch_size)
        if capacity <= 0:
            raise ValueError("batch_size must be positive.")
        source_seed = int(seed)
        if source_seed < 0:
            raise ValueError("seed must be nonnegative.")
        batches = IndexEpochPlan(
            population,
            capacity,
            True,
            source_seed,
            0,
            False,
        ).batch_count
        configuration = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "num_factors": population,
            "batch_size": capacity,
            "seed": source_seed,
            "ordering": EPOCH_ORDER_ALGORITHM,
            "data": array_tree_fingerprint(arrays),
        }
        configuration_json = json.dumps(
            configuration,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.data = arrays
        self._num_factors = population
        self._batch_capacity = capacity
        self._seed = source_seed
        self._batches_per_epoch = batches
        self._configuration_json = configuration_json
        self._fingerprint = hashlib.sha256(configuration_json.encode("utf-8")).hexdigest()

    @property
    def num_factors(self) -> int:
        return self._num_factors

    @property
    def batch_capacity(self) -> int:
        return self._batch_capacity

    @property
    def batches_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def configuration(self) -> Mapping[str, Any]:
        return json.loads(self._configuration_json)

    def epoch(self, epoch: int, /) -> Iterator[LikelihoodBatch]:
        plan = IndexEpochPlan(
            self._num_factors,
            self._batch_capacity,
            True,
            self._seed,
            int(epoch),
            False,
        )
        for _, active_indices in plan.iter_batches():
            yield self._batch(active_indices, audit=False)

    def audit_epoch(self) -> Iterator[LikelihoodBatch]:
        """Enumerate every factor exactly once with unit estimator weights."""
        plan = IndexEpochPlan(
            self._num_factors,
            self._batch_capacity,
            False,
            self._seed,
            0,
            False,
        )
        for _, active_indices in plan.iter_batches():
            yield self._batch(active_indices, audit=True)

    def _batch(
        self, active_indices: tuple[int, ...], /, *, audit: bool
    ) -> LikelihoodBatch:
        active_count = len(active_indices)
        padded_indices = active_indices + (active_indices[-1],) * (
            self._batch_capacity - active_count
        )
        batch_indices = jnp.asarray(padded_indices, dtype=jnp.int32)
        batch_data = jax.tree_util.tree_map(
            lambda leaf: leaf[batch_indices],
            self.data,
        )
        factor_mask = jnp.arange(self._batch_capacity) < active_count
        probabilities = jnp.full(
            (self._batch_capacity,),
            1.0 / self._num_factors,
            dtype=jnp.result_type(float),
        )
        weights = jnp.full(
            (self._batch_capacity,),
            1.0 if audit else self._num_factors / active_count,
            dtype=probabilities.dtype,
        )
        return LikelihoodBatch(
            batch_data,
            factor_mask,
            factor_ids=batch_indices,
            sampling_probabilities=probabilities,
            estimator_weights=weights,
        )


class ImportanceMinibatchSource(StrictModule):
    """IID nonuniform-with-replacement source with exact importance weights."""

    data: PyTree[Array]
    probabilities: Array
    root_key: Array
    _num_factors: int = eqx.field(static=True)
    _batch_capacity: int = eqx.field(static=True)
    _batches_per_epoch: int = eqx.field(static=True)
    probability_epoch: int = eqx.field(static=True)
    epoch_span: int = eqx.field(static=True)
    _configuration_json: str = eqx.field(static=True)
    _fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        data: PyTree[ArrayLike],
        probabilities: ArrayLike,
        /,
        *,
        batch_size: int,
        seed: int = 0,
        probability_epoch: int = 0,
        epoch_span: int = 1,
    ):
        arrays = jax.tree_util.tree_map(jnp.asarray, data)
        leaves = jax.tree_util.tree_leaves(arrays)
        if not leaves:
            raise ValueError("Importance source data must contain array leaves.")
        population = int(leaves[0].shape[0]) if leaves[0].ndim else 0
        if population <= 0 or any(
            leaf.ndim == 0 or int(leaf.shape[0]) != population for leaf in leaves
        ):
            raise ValueError("All source leaves must share a positive leading axis.")
        probability_array = jnp.asarray(probabilities, dtype=float)
        if probability_array.shape != (population,):
            raise ValueError("probabilities must have one entry per source factor.")
        if bool(jnp.any(~jnp.isfinite(probability_array))) or bool(
            jnp.any(probability_array <= 0.0)
        ):
            raise ValueError("Importance probabilities must be finite and positive.")
        if not bool(
            jnp.isclose(
                jnp.sum(probability_array),
                jnp.asarray(1.0, dtype=probability_array.dtype),
            )
        ):
            raise ValueError("Importance probabilities must sum to one.")
        capacity = int(batch_size)
        source_seed = int(seed)
        probability_epoch_ = int(probability_epoch)
        span = int(epoch_span)
        if capacity <= 0 or source_seed < 0 or probability_epoch_ < 0 or span <= 0:
            raise ValueError(
                "batch_size/epoch_span must be positive and indices nonnegative."
            )
        batches = (population + capacity - 1) // capacity
        configuration = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "num_factors": population,
            "batch_size": capacity,
            "seed": source_seed,
            "probability_epoch": probability_epoch_,
            "epoch_span": span,
            "probabilities": array_tree_fingerprint(probability_array),
            "data": array_tree_fingerprint(arrays),
            "design": "iid-nonuniform-with-replacement",
        }
        configuration_json = json.dumps(
            configuration,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.data = arrays
        self.probabilities = probability_array
        self.root_key = jr.key(source_seed)
        self._num_factors = population
        self._batch_capacity = capacity
        self._batches_per_epoch = batches
        self.probability_epoch = probability_epoch_
        self.epoch_span = span
        self._configuration_json = configuration_json
        self._fingerprint = hashlib.sha256(configuration_json.encode("utf-8")).hexdigest()

    @property
    def num_factors(self) -> int:
        return self._num_factors

    @property
    def batch_capacity(self) -> int:
        return self._batch_capacity

    @property
    def batches_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def configuration(self) -> Mapping[str, Any]:
        return json.loads(self._configuration_json)

    def epoch(self, epoch: int, /) -> Iterator[LikelihoodBatch]:
        epoch_index = int(epoch)
        if epoch_index < 0:
            raise ValueError("epoch must be nonnegative.")
        if epoch_index >= self.epoch_span:
            raise ValueError(
                "Importance probabilities are stale beyond the declared epoch_span."
            )
        address = SampleAddress(
            "uq.minibatch",
            "importance-selection",
            target=self.fingerprint,
            role="factor",
        )
        for batch_index in range(self.batches_per_epoch):
            key = derive_key(
                self.root_key,
                address,
                self.probability_epoch,
                epoch_index,
                batch_index,
            )
            ids = jr.choice(
                key,
                self.num_factors,
                shape=(self.batch_capacity,),
                replace=True,
                p=self.probabilities,
            )
            selected_probabilities = self.probabilities[ids]
            yield LikelihoodBatch(
                jax.tree_util.tree_map(lambda leaf: leaf[ids], self.data),
                jnp.ones((self.batch_capacity,), dtype=bool),
                factor_ids=ids,
                sampling_probabilities=selected_probabilities,
                estimator_weights=1.0 / (self.batch_capacity * selected_probabilities),
            )

    def audit_epoch(self) -> Iterator[LikelihoodBatch]:
        source = ArrayMinibatchSource(
            self.data,
            batch_size=self.batch_capacity,
            seed=0,
        )
        yield from source.audit_epoch()


class MinibatchPosteriorProblem(StrictModule):
    """Factorized posterior with unbiased minibatch likelihood estimators."""

    parameter_space: ParameterSpace
    observation_factor: AbstractObservationFactor | None
    log_likelihood_factors_fn: Callable[[PyTree[Any], LikelihoodBatch], ArrayLike]
    num_factors: int = eqx.field(static=True)
    full_log_likelihood_fn: Callable[[PyTree[Any]], ArrayLike] | None = eqx.field(
        static=True
    )
    predict_fn: Callable[..., Any] | None = eqx.field(static=True)
    observation_variance_fn: Callable[..., Any] | None = eqx.field(static=True)
    sample_observation_fn: Callable[..., Any] | None = eqx.field(static=True)

    def __init__(
        self,
        parameter_space: ParameterSpace,
        log_likelihood_factors: AbstractObservationFactor
        | Callable[[PyTree[Any], LikelihoodBatch], ArrayLike],
        /,
        *,
        num_factors: int,
        full_log_likelihood: Callable[[PyTree[Any]], ArrayLike] | None = None,
        predict: Callable[..., Any] | None = None,
        observation_variance: Callable[..., Any] | None = None,
        sample_observation: Callable[..., Any] | None = None,
    ):
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        if not isinstance(
            log_likelihood_factors, AbstractObservationFactor
        ) and not callable(log_likelihood_factors):
            raise TypeError(
                "log_likelihood_factors must be an AbstractObservationFactor or callable."
            )
        count = int(num_factors)
        if count <= 0:
            raise ValueError("num_factors must be positive.")
        for name, function in (
            ("full_log_likelihood", full_log_likelihood),
            ("predict", predict),
            ("observation_variance", observation_variance),
            ("sample_observation", sample_observation),
        ):
            if function is not None and not callable(function):
                raise TypeError(f"{name} must be callable or None.")
        self.parameter_space = parameter_space
        self.observation_factor = (
            log_likelihood_factors
            if isinstance(log_likelihood_factors, AbstractObservationFactor)
            else None
        )
        if self.observation_factor is not None:
            if self.observation_factor.semantics != "normalized_likelihood":
                raise ValueError(
                    "Posterior minibatches require normalized_likelihood semantics."
                )
            self.log_likelihood_factors_fn = self.observation_factor.log_factors
        else:
            self.log_likelihood_factors_fn = log_likelihood_factors
        self.num_factors = count
        self.full_log_likelihood_fn = full_log_likelihood
        self.predict_fn = predict
        self.observation_variance_fn = observation_variance
        self.sample_observation_fn = sample_observation

    @property
    def initial_position(self) -> PyTree[Any]:
        return self.parameter_space.initial

    def log_likelihood_factors(
        self,
        physical: PyTree[Any],
        batch: LikelihoodBatch,
        /,
    ) -> Array:
        if not isinstance(batch, LikelihoodBatch):
            raise TypeError("batch must be a LikelihoodBatch.")
        factors = jnp.asarray(self.log_likelihood_factors_fn(physical, batch))
        if not jnp.issubdtype(factors.dtype, jnp.floating):
            raise TypeError("Observation log factors must have real floating dtype.")
        if factors.ndim != 1 or factors.shape != batch.factor_mask.shape:
            raise ValueError(
                "log_likelihood_factors must return one scalar per batch capacity."
            )
        return jnp.where(batch.factor_mask, factors, jnp.zeros_like(factors))

    def log_likelihood_estimate(
        self,
        physical: PyTree[Any],
        batch: LikelihoodBatch,
        /,
    ) -> Array:
        factors = self.log_likelihood_factors(physical, batch)
        return jnp.sum(batch.estimator_weights * factors).reshape(())

    def log_density_estimate(
        self,
        position: PyTree[Any],
        batch: LikelihoodBatch,
        /,
    ) -> Array:
        physical = self.parameter_space.constrain(position)
        return (
            self.log_likelihood_estimate(physical, batch)
            + self.parameter_space.log_prior(physical)
            + self.parameter_space.log_abs_det_jacobian(position)
        )

    def full_log_likelihood(self, physical: PyTree[Any], /) -> Array:
        if self.full_log_likelihood_fn is None:
            raise ValueError(
                "MinibatchPosteriorProblem has no full log-likelihood function."
            )
        value = jnp.asarray(self.full_log_likelihood_fn(physical), dtype=float)
        if value.ndim != 0:
            raise ValueError("full_log_likelihood must return a scalar.")
        return value

    def full_log_density(self, position: PyTree[Any], /) -> Array:
        physical = self.parameter_space.constrain(position)
        return (
            self.full_log_likelihood(physical)
            + self.parameter_space.log_prior(physical)
            + self.parameter_space.log_abs_det_jacobian(position)
        )

    def predict(self, position: PyTree[Any], /, *args: Any, **kwargs: Any) -> Any:
        if self.predict_fn is None:
            raise ValueError("MinibatchPosteriorProblem has no prediction function.")
        physical = self.parameter_space.constrain(position)
        return self.predict_fn(physical, *args, **kwargs)

    def conditional_observation_variance(
        self,
        position: PyTree[Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.observation_variance_fn is None:
            raise ValueError(
                "MinibatchPosteriorProblem has no observation-variance function."
            )
        physical = self.parameter_space.constrain(position)
        return self.observation_variance_fn(physical, *args, **kwargs)

    def sample_observation(
        self,
        key: Any,
        position: PyTree[Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.sample_observation_fn is None:
            raise ValueError(
                "MinibatchPosteriorProblem has no observation-sampling function."
            )
        physical = self.parameter_space.constrain(position)
        return self.sample_observation_fn(key, physical, *args, **kwargs)

    def validate(self, batch: LikelihoodBatch, /) -> tuple[Array, PyTree[Any]]:
        value, gradient = jax.value_and_grad(self.log_density_estimate)(
            self.initial_position,
            batch,
        )
        if value.ndim != 0 or not bool(jnp.isfinite(value)):
            raise FloatingPointError("Initial stochastic log density must be finite.")
        if any(
            bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(gradient)
        ):
            raise FloatingPointError("Initial stochastic gradient must be finite.")
        return value, gradient


def prepare_importance_minibatch_source(
    problem: MinibatchPosteriorProblem,
    base_source: ArrayMinibatchSource,
    anchor_position: PyTree[Any],
    /,
    *,
    score: Literal["absolute_log_likelihood", "gradient_norm"] = (
        "absolute_log_likelihood"
    ),
    uniform_mixture: float,
    epoch_span: int,
    seed: int = 0,
    probability_epoch: int = 0,
) -> ImportanceMinibatchSource:
    """Freeze positive factor probabilities at a declared anchor position."""
    if not isinstance(problem, MinibatchPosteriorProblem):
        raise TypeError("problem must be a MinibatchPosteriorProblem.")
    if not isinstance(base_source, ArrayMinibatchSource):
        raise TypeError("base_source must be an ArrayMinibatchSource.")
    if problem.num_factors != base_source.num_factors:
        raise ValueError("Problem and source factor populations must match.")
    mixture = float(uniform_mixture)
    if not 0.0 < mixture <= 1.0:
        raise ValueError("uniform_mixture must lie in (0, 1].")
    if score not in ("absolute_log_likelihood", "gradient_norm"):
        raise ValueError("Unknown importance score.")
    physical = problem.parameter_space.constrain(anchor_position)
    scores = jnp.zeros((base_source.num_factors,), dtype=float)
    for batch in base_source.audit_epoch():
        if score == "absolute_log_likelihood":
            batch_scores = jnp.abs(problem.log_likelihood_factors(physical, batch))
        else:
            jacobian = jax.jacrev(problem.log_likelihood_factors)(physical, batch)
            squared = jnp.zeros(batch.factor_mask.shape, dtype=float)
            for leaf in jax.tree_util.tree_leaves(jacobian):
                leaf_array = jnp.asarray(leaf)
                squared = squared + jnp.sum(
                    jnp.square(jnp.abs(leaf_array)),
                    axis=tuple(range(1, leaf_array.ndim)),
                )
            batch_scores = jnp.sqrt(squared)
        active_ids = jnp.where(batch.factor_mask, batch.factor_ids, 0)
        scores = scores.at[active_ids].add(
            jnp.where(batch.factor_mask, batch_scores, 0.0)
        )
    if bool(jnp.any(~jnp.isfinite(scores))) or bool(jnp.all(scores == 0.0)):
        raise ValueError("Importance scores must be finite with positive total mass.")
    probabilities = (1.0 - mixture) * scores / jnp.sum(
        scores
    ) + mixture / base_source.num_factors
    return ImportanceMinibatchSource(
        base_source.data,
        probabilities,
        batch_size=base_source.batch_capacity,
        seed=seed,
        probability_epoch=probability_epoch,
        epoch_span=epoch_span,
    )


__all__ = [
    "AbstractObservationFactor",
    "ArrayMinibatchSource",
    "FactorSamplingState",
    "ImportanceMinibatchSource",
    "LikelihoodBatch",
    "MinibatchPosteriorProblem",
    "MinibatchSource",
    "ObservationFactorSemantics",
    "prepare_importance_minibatch_source",
]
