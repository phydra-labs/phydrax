#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._likelihoods import AbstractLikelihood
from ..stochastic._state_space import AbstractObservationModel, StateSpaceStepContext


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError(f"{owner} dimensions must be positive.")
    return resolved


class LikelihoodObservationModel(AbstractObservationModel):
    """Adapt an elementwise Phydrax likelihood to a state-space observation law."""

    likelihood: AbstractLikelihood
    location_fn: Callable[[Array, Array, StateSpaceStepContext], Array]
    parameters_fn: (
        Callable[[Array, Array, StateSpaceStepContext], Mapping[str, Any]] | None
    )
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        likelihood: AbstractLikelihood,
        location: Callable[[Array, Array, StateSpaceStepContext], Array],
        /,
        *,
        state_shape: Sequence[int],
        observation_shape: Sequence[int],
        observation_id: str,
        parameters: (
            Callable[[Array, Array, StateSpaceStepContext], Mapping[str, Any]] | None
        ) = None,
    ):
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if not callable(location):
            raise TypeError("location must be callable.")
        if parameters is not None and not callable(parameters):
            raise TypeError("parameters must be callable or None.")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be a non-empty string.")
        self.likelihood = likelihood
        self.location_fn = location
        self.parameters_fn = parameters
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.observation_shape = _shape(observation_shape, owner="observation_shape")
        self.observation_id = observation_id

    def _parameters(
        self, state: Array, time: Array, context: StateSpaceStepContext, /
    ) -> Mapping[str, Any]:
        return (
            {} if self.parameters_fn is None else self.parameters_fn(state, time, context)
        )

    def location(
        self,
        state: ArrayLike,
        time: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        values = jnp.asarray(self.location_fn(state_array, jnp.asarray(time), context))
        if (
            self.observation_shape
            and tuple(values.shape[-len(self.observation_shape) :])
            != self.observation_shape
        ):
            raise ValueError("Observation location has an incompatible trailing shape.")
        return values

    def log_prob(
        self,
        value: ArrayLike,
        state: ArrayLike,
        time: ArrayLike,
        mask: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        time_array = jnp.asarray(time)
        location = self.location(state_array, time_array, context)
        target = jnp.broadcast_to(jnp.asarray(value), location.shape)
        active = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), location.shape)
        terms = jnp.asarray(
            self.likelihood.log_prob(
                location,
                target,
                **self._parameters(state_array, time_array, context),
            )
        )
        terms = jnp.broadcast_to(terms, location.shape)
        if not self.observation_shape:
            return jnp.where(active, terms, 0.0)
        axes = tuple(range(terms.ndim - len(self.observation_shape), terms.ndim))
        return jnp.sum(jnp.where(active, terms, 0.0), axis=axes)

    def sample(
        self,
        key: Key[Array, ""],
        state: ArrayLike,
        time: ArrayLike,
        context: StateSpaceStepContext,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        state_array = jnp.asarray(state)
        time_array = jnp.asarray(time)
        location = self.location(state_array, time_array, context)
        parameters = self._parameters(state_array, time_array, context)
        shape = _shape(sample_shape, owner="sample_shape")
        if not shape:
            return jnp.asarray(self.likelihood.sample(key, location, **parameters))
        count = prod(shape)
        keys = jr.split(key, count)
        values = jax.vmap(
            lambda sample_key: self.likelihood.sample(sample_key, location, **parameters)
        )(keys)
        return values.reshape(shape + location.shape)


__all__ = ["LikelihoodObservationModel"]
