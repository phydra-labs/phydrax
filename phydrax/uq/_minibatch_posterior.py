#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._data_plane import EPOCH_ORDER_ALGORITHM, IndexEpochPlan
from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from ._posterior import ParameterSpace


class LikelihoodBatch(StrictModule):
    """Fixed-capacity likelihood data and its active-factor mask."""

    data: PyTree[Any]
    factor_mask: Array

    def __init__(self, data: PyTree[Any], factor_mask: ArrayLike, /):
        mask = jnp.asarray(factor_mask)
        if mask.ndim != 1:
            raise ValueError("factor_mask must be one-dimensional.")
        if mask.dtype != jnp.bool_:
            raise TypeError("factor_mask must have boolean dtype.")
        if mask.shape[0] == 0:
            raise ValueError("factor_mask must have positive capacity.")
        if not bool(jnp.any(mask)):
            raise ValueError("LikelihoodBatch must contain at least one active factor.")
        self.data = data
        self.factor_mask = mask

    @property
    def capacity(self) -> int:
        return int(self.factor_mask.shape[0])

    @property
    def factor_count(self) -> Array:
        return jnp.sum(self.factor_mask, dtype=jnp.int32)


@runtime_checkable
class MinibatchSource(Protocol):
    """Deterministic, finite sequence of uniformly sampled likelihood batches."""

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
            yield LikelihoodBatch(batch_data, factor_mask)


class MinibatchPosteriorProblem(StrictModule):
    """Factorized posterior with unbiased minibatch likelihood estimators."""

    parameter_space: ParameterSpace
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
        log_likelihood_factors: Callable[[PyTree[Any], LikelihoodBatch], ArrayLike],
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
        if not callable(log_likelihood_factors):
            raise TypeError("log_likelihood_factors must be callable.")
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
        factors = jnp.asarray(
            self.log_likelihood_factors_fn(physical, batch),
            dtype=float,
        )
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
        active_count = jnp.sum(batch.factor_mask, dtype=factors.dtype)
        return jnp.asarray(self.num_factors, dtype=factors.dtype) * (
            jnp.sum(factors) / active_count
        )

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


__all__ = [
    "ArrayMinibatchSource",
    "LikelihoodBatch",
    "MinibatchPosteriorProblem",
    "MinibatchSource",
]
