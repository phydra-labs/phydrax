#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._strict import StrictModule
from .._trainable import NonTrainableState


class AbstractDistributionEvolutionMethod(abc.ABC):
    """Native ask/tell contract for distribution-based functional search."""

    population_size: int

    @abc.abstractmethod
    def init(self, key: Key, mean: Any, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def ask(self, key: Key, state: Any, /) -> tuple[Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def tell(self, population: Any, losses: Array, state: Any, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, state: Any, /) -> Any:
        raise NotImplementedError


class OpenEvolutionState(StrictModule):
    mean: Any
    standard_deviation: Array
    generation: Array


class OpenEvolutionStrategy(
    StrictModule,
    NonTrainableState,
    AbstractDistributionEvolutionMethod,
):
    """Antithetic rank-normalized OpenAI-style evolution strategy."""

    population_size: int = eqx.field(static=True)
    initial_standard_deviation: float = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    standard_deviation_decay: float = eqx.field(static=True)
    minimum_standard_deviation: float = eqx.field(static=True)

    def __init__(
        self,
        population_size: int,
        /,
        *,
        initial_standard_deviation: float = 0.1,
        learning_rate: float = 0.05,
        standard_deviation_decay: float = 1.0,
        minimum_standard_deviation: float = 1.0e-6,
    ):
        population = int(population_size)
        deviation = float(initial_standard_deviation)
        rate = float(learning_rate)
        decay = float(standard_deviation_decay)
        minimum = float(minimum_standard_deviation)
        if population < 2 or population % 2:
            raise ValueError("population_size must be a positive even integer >= 2.")
        if deviation <= 0.0 or rate <= 0.0 or minimum <= 0.0:
            raise ValueError(
                "Evolution scale, learning rate, and minimum scale must be positive."
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("standard_deviation_decay must lie in (0, 1].")
        if minimum > deviation:
            raise ValueError(
                "minimum_standard_deviation cannot exceed the initial value."
            )
        self.population_size = population
        self.initial_standard_deviation = deviation
        self.learning_rate = rate
        self.standard_deviation_decay = decay
        self.minimum_standard_deviation = minimum

    def init(self, key: Key, mean: Any, /) -> OpenEvolutionState:
        del key
        leaves = jax.tree_util.tree_leaves(mean)
        if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
            raise TypeError("Evolution means must be nonempty PyTrees of inexact arrays.")
        dtype = jnp.result_type(*(leaf.dtype for leaf in leaves))
        return OpenEvolutionState(
            mean,
            jnp.asarray(self.initial_standard_deviation, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def ask(
        self,
        key: Key,
        state: OpenEvolutionState,
        /,
    ) -> tuple[Any, OpenEvolutionState]:
        leaves, structure = jax.tree_util.tree_flatten(state.mean)
        keys = jax.random.split(key, len(leaves))
        half = self.population_size // 2
        population_leaves = []
        for leaf, leaf_key in zip(leaves, keys, strict=True):
            noise_half = jax.random.normal(
                leaf_key,
                (half, *leaf.shape),
                dtype=leaf.dtype,
            )
            noise = jnp.concatenate((noise_half, -noise_half), axis=0)
            population_leaves.append(leaf[None, ...] + state.standard_deviation * noise)
        return jax.tree_util.tree_unflatten(structure, population_leaves), state

    def tell(
        self,
        population: Any,
        losses: Array,
        state: OpenEvolutionState,
        /,
    ) -> OpenEvolutionState:
        losses_ = jnp.asarray(losses)
        if losses_.shape != (self.population_size,):
            raise ValueError("losses must have one scalar per population member.")
        finite = jnp.isfinite(losses_)
        any_finite = jnp.any(finite)
        maximum = jnp.max(jnp.where(finite, losses_, -jnp.inf))
        ranked = jnp.where(finite, losses_, maximum + 1.0)
        centered = ranked - jnp.mean(ranked)
        scale = jnp.std(ranked)
        utilities = -centered / jnp.where(scale > 0.0, scale, 1.0)
        utilities = jnp.where(any_finite, utilities, jnp.zeros_like(utilities))
        normalizer = jnp.maximum(jnp.sum(jnp.abs(utilities)), 1.0)
        utilities = utilities / normalizer

        def update(mean_leaf, population_leaf):
            noise = (population_leaf - mean_leaf[None, ...]) / state.standard_deviation
            direction = jnp.tensordot(utilities, noise, axes=((0,), (0,)))
            return mean_leaf + self.learning_rate * direction

        mean = jax.tree.map(update, state.mean, population)
        deviation = jnp.maximum(
            state.standard_deviation * self.standard_deviation_decay,
            self.minimum_standard_deviation,
        )
        return OpenEvolutionState(mean, deviation, state.generation + 1)

    def mean(self, state: OpenEvolutionState, /) -> Any:
        return state.mean


__all__ = [
    "AbstractDistributionEvolutionMethod",
    "OpenEvolutionState",
    "OpenEvolutionStrategy",
]
