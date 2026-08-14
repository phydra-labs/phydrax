#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule
from ._stochastic_estimators import _directional_second_derivative


DimensionSamplingMode: TypeAlias = Literal["uniform", "importance"]


def _policy_id(parts: tuple[object, ...], /) -> str:
    digest = sha256(b"phydrax-dimension-sampling\0")
    digest.update(repr(parts).encode("utf-8"))
    return digest.hexdigest()


class DimensionSamplingPolicy(StrictModule):
    """Coordinate-subset policy for an unbiased finite-sum estimate."""

    probabilities: Array | None
    total_dimension: int = eqx.field(static=True)
    subset_size: int = eqx.field(static=True)
    sampling: DimensionSamplingMode = eqx.field(static=True)
    replace: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_dimension: int,
        subset_size: int,
        /,
        *,
        sampling: DimensionSamplingMode = "uniform",
        replace: bool = False,
        probabilities: ArrayLike | None = None,
        policy_id: str | None = None,
    ):
        dimension = int(total_dimension)
        count = int(subset_size)
        if dimension < 1 or count < 1:
            raise ValueError("total_dimension and subset_size must be positive.")
        replacement = bool(replace)
        if not replacement and count > dimension:
            raise ValueError("subset_size cannot exceed dimension without replacement.")
        if sampling not in ("uniform", "importance"):
            raise ValueError("sampling must be 'uniform' or 'importance'.")
        if sampling == "uniform":
            if probabilities is not None:
                raise ValueError(
                    "Uniform dimension sampling does not accept probabilities."
                )
            probs = None
            probability_identity = None
        else:
            if not replacement:
                raise ValueError("Importance sampling currently requires replacement.")
            if probabilities is None:
                raise ValueError("Importance sampling requires probabilities.")
            probs = jnp.asarray(probabilities, dtype=float).reshape((-1,))
            if probs.shape != (dimension,):
                raise ValueError("probabilities must have shape (total_dimension,).")
            if bool(jnp.any(~jnp.isfinite(probs))) or bool(jnp.any(probs <= 0.0)):
                raise ValueError("probabilities must be finite and strictly positive.")
            if not bool(jnp.isclose(jnp.sum(probs), 1.0)):
                raise ValueError("probabilities must sum to one.")
            probability_identity = tuple(float(value) for value in probs)
        identity = (
            dimension,
            count,
            sampling,
            replacement,
            probability_identity,
        )
        self.probabilities = probs
        self.total_dimension = dimension
        self.subset_size = count
        self.sampling = sampling
        self.replace = replacement
        self.policy_id = _policy_id(identity) if policy_id is None else str(policy_id)
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty.")


class DimensionOperatorEstimate(StrictModule):
    value: Array
    standard_error: Array
    total_dimension: int = eqx.field(static=True)
    subset_size: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


class DimensionOperatorSamples(StrictModule):
    """Scaled coordinate realizations of one unbiased finite-sum estimator."""

    indices: Array
    values: Array
    mean: Array
    sample_variance: Array
    standard_error: Array
    dependence_ids: Array
    total_dimension: int = eqx.field(static=True)
    subset_size: int = eqx.field(static=True)
    sampling: DimensionSamplingMode = eqx.field(static=True)
    replace: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        indices: ArrayLike,
        values: ArrayLike,
        policy: DimensionSamplingPolicy,
        /,
    ):
        if not isinstance(policy, DimensionSamplingPolicy):
            raise TypeError("policy must be a DimensionSamplingPolicy.")
        sampled_indices = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        samples = jnp.asarray(values)
        if sampled_indices.shape != (policy.subset_size,):
            raise ValueError("indices must have shape (subset_size,).")
        if samples.shape[0] != policy.subset_size:
            raise ValueError("values must have subset_size as their first axis.")
        mean = jnp.mean(samples, axis=0)
        if policy.subset_size == 1:
            sample_variance = jnp.full(mean.shape, jnp.nan, dtype=float)
            standard_error = jnp.full(mean.shape, jnp.nan, dtype=float)
        else:
            centered = samples - mean
            sample_variance = jnp.sum(jnp.abs(centered) ** 2, axis=0) / float(
                policy.subset_size - 1
            )
            correction = (
                1.0 - policy.subset_size / float(policy.total_dimension)
                if policy.sampling == "uniform" and not policy.replace
                else 1.0
            )
            standard_error = jnp.sqrt(
                correction * sample_variance / float(policy.subset_size)
            )
        self.indices = sampled_indices
        self.values = samples
        self.mean = mean
        self.sample_variance = sample_variance
        self.standard_error = standard_error
        self.dependence_ids = jnp.arange(policy.subset_size, dtype=jnp.int32)
        self.total_dimension = policy.total_dimension
        self.subset_size = policy.subset_size
        self.sampling = policy.sampling
        self.replace = policy.replace
        self.policy_id = policy.policy_id

    def estimate(self) -> DimensionOperatorEstimate:
        return DimensionOperatorEstimate(
            self.mean,
            self.standard_error,
            self.total_dimension,
            self.subset_size,
            self.policy_id,
        )


def _sample_indices(
    key: Key[Array, ""],
    policy: DimensionSamplingPolicy,
    /,
) -> Array:
    probabilities = None if policy.sampling == "uniform" else policy.probabilities
    return jr.choice(
        key,
        policy.total_dimension,
        shape=(policy.subset_size,),
        replace=policy.replace,
        p=probabilities,
    )


def dimension_sum_samples(
    contribution: Callable[[Array], Array],
    key: Key[Array, ""],
    policy: DimensionSamplingPolicy,
    /,
) -> DimensionOperatorSamples:
    """Sample coordinate contributions and return an unbiased sum estimator."""
    if not callable(contribution):
        raise TypeError("contribution must be callable.")
    if not isinstance(policy, DimensionSamplingPolicy):
        raise TypeError("policy must be a DimensionSamplingPolicy.")
    indices = _sample_indices(key, policy)
    values = jax.vmap(contribution)(indices)
    if policy.sampling == "uniform":
        scaled = float(policy.total_dimension) * values
    else:
        if policy.probabilities is None:
            raise RuntimeError("Importance probabilities are unavailable.")
        selected = policy.probabilities[indices]
        scaled = values / selected.reshape(selected.shape + (1,) * (values.ndim - 1))
    return DimensionOperatorSamples(indices, scaled, policy)


def estimate_dimension_sum(
    contribution: Callable[[Array], Array],
    key: Key[Array, ""],
    policy: DimensionSamplingPolicy,
    /,
) -> DimensionOperatorEstimate:
    return dimension_sum_samples(contribution, key, policy).estimate()


def coordinate_divergence_samples(
    vector_field: Callable[[Array], Array],
    state: ArrayLike,
    key: Key[Array, ""],
    policy: DimensionSamplingPolicy,
    /,
) -> DimensionOperatorSamples:
    """Estimate a coordinate divergence by sampling Jacobian diagonal entries."""
    state_array = jnp.asarray(state)
    if state_array.ndim < 1:
        raise ValueError("state must have at least one axis.")
    state_size = prod(state_array.shape)
    if state_size != policy.total_dimension:
        raise ValueError(
            "policy.total_dimension must equal the flattened state dimension."
        )
    prototype = jnp.asarray(vector_field(state_array))
    if prototype.shape != state_array.shape:
        raise ValueError("vector_field must preserve the state shape.")

    def contribution(index):
        direction = jax.nn.one_hot(
            index,
            state_size,
            dtype=state_array.dtype,
        ).reshape(state_array.shape)
        _, derivative = jax.jvp(vector_field, (state_array,), (direction,))
        return jnp.asarray(derivative).reshape((-1,))[index]

    return dimension_sum_samples(contribution, key, policy)


def coordinate_second_derivative_samples(
    function: Callable[[Array], Array],
    state: ArrayLike,
    key: Key[Array, ""],
    policy: DimensionSamplingPolicy,
    /,
) -> DimensionOperatorSamples:
    """Estimate a Laplacian by sampling diagonal Hessian contributions."""
    state_array = jnp.asarray(state)
    if state_array.ndim < 1:
        raise ValueError("state must have at least one axis.")
    state_size = prod(state_array.shape)
    if state_size != policy.total_dimension:
        raise ValueError(
            "policy.total_dimension must equal the flattened state dimension."
        )

    def contribution(index):
        direction = jax.nn.one_hot(
            index,
            state_size,
            dtype=state_array.dtype,
        ).reshape(state_array.shape)
        return _directional_second_derivative(
            function,
            state_array,
            direction,
            direction,
        )

    return dimension_sum_samples(contribution, key, policy)


__all__ = [
    "coordinate_divergence_samples",
    "coordinate_second_derivative_samples",
    "DimensionOperatorEstimate",
    "DimensionOperatorSamples",
    "DimensionSamplingMode",
    "DimensionSamplingPolicy",
    "dimension_sum_samples",
    "estimate_dimension_sum",
]
